import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import openai

import Py_files.twcs_processor as processor
from Py_files.llm_extractor import LLMExtractor
from Notebooks.VectorDBStructure.query import query_similar
from Notebooks.VectorDBStructure.db_structure import DatabaseStructure
from Py_files.reranker import CrossEncoderReranker
from Py_files.prompts import ENDBOT_PROMPT

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ChatQAPipeline:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.db = DatabaseStructure()
        self.reranker = CrossEncoderReranker(top_k=50)

    def run(self, user_input: str) -> str:
        """Returns only the final LLM response."""
        response, _ = self.run_with_payload(user_input)
        return response

    def get_rag_payload(self, user_input: str) -> str:
        """Returns only the RAG payload as JSON string."""
        _, rag_payload = self.run_with_payload(user_input)
        return rag_payload

    def run_with_payload(self, user_input: str) -> Tuple[str, str]:
        """
        Runs the whole pipeline once and returns both the final LLM response and the RAG payload.
        Returns:
            (llm_response: str, rag_payload: str)
        """
        
        cleaned_input = self._preprocess_input(user_input)
        entities,  relationship = self._extract_entities_and_relationships(cleaned_input)
        embedding = self._create_embedding(conversation,entities, relationship)
        hits = self._retrieve_similar(embedding)
        candidates = self._extract_candidates(hits)
        reranked = self._rerank(user_input, candidates)
        rows = self._build_rows(user_input, hits, reranked)
        reranked_qa = self._normalize_and_score(rows)
        top5_df = self._select_top5(reranked_qa)
        rag_payload = self._build_payload_per_qa(top5_df, user_input)
        response = self._query_llm(rag_payload)
        return response, rag_payload

    # --- Pipeline Step Functions ---

    def _preprocess_input(self, user_input: str) -> str:
        cleaned = processor.TWCSProcessor._clean_single(user_input)
        return processor.TWCSProcessor._convert_to_conversation(cleaned)

    def _extract_entities_and_relationships(self, cleaned_input: str):
        df = pd.DataFrame([[cleaned_input]], columns=['cleaned_conversations'])
        pipe = LLMExtractor(dataframe=df)
        _ = pipe.extract_entities()
        _ = pipe.process_entities_json()
        df_final = pipe.extract_relationships()
        entities = df_final['entities'].values[0]
        relationship = df_final['relationship'].values[0]
        return entities,relationship

    def _create_embedding(self, conversation,entities,relationship):
        fixed_relationships = self.db.fix_relationships(relationship)
        return self.db.text_to_embedding(conversation,entities, fixed_relationships).tolist()

    def _retrieve_similar(self, embedding):
        return query_similar(embedding, k=50)

    def _extract_candidates(self, hits):
        return [hit["_source"]["Conversation_History"]["conversation"] for hit in hits]

    def _rerank(self, user_input, candidates):
        return self.reranker.rerank(user_input, candidates)

    def _build_rows(self, user_input, hits, reranked):
        score_rank_map = {
            conv: (score, rank + 1)
            for rank, (conv, score) in enumerate(reranked)
        }
        rows = []
        for hit in hits:
            src = hit["_source"]
            conv = src["Conversation_History"]["conversation"]
            score, rank = score_rank_map.get(conv, (0.0, None))
            rows.append({
                "prompt": user_input,
                "id": hit["_id"],
                "similarity_score": hit["_score"],
                "rerank_score": score,
                "rerank_rank": rank,
                "ChatID": src["ChatID"],
                "Company_name": src["Company_name"],
                "Conversation_History": conv,
                "Entities": json.dumps(src["Entities"]),
                "Relationships": json.dumps(src["Relationships"])
            })
        return rows

    def _normalize_and_score(self, rows):
        reranked_qa = pd.DataFrame(rows).sort_values(by="rerank_rank", na_position="last").reset_index(drop=True)
        scaler = StandardScaler()
        reranked_qa[["sim_norm", "rerank_norm"]] = scaler.fit_transform(
            reranked_qa[["similarity_score", "rerank_score"]].fillna(0)
        )
        reranked_qa["hybrid_score"] = 0.3 * reranked_qa["sim_norm"] + 0.7 * reranked_qa["rerank_norm"]
        return reranked_qa

    def _select_top5(self, reranked_qa):
        top_candidates = reranked_qa.sort_values(by="hybrid_score", ascending=False).head(10)
        seen_combinations = set()
        filtered_rows = []
        for _, row in top_candidates.iterrows():
            key = (row["Entities"], row["Relationships"])
            if key not in seen_combinations:
                filtered_rows.append(row)
                seen_combinations.add(key)
            if len(filtered_rows) == 5:
                break
        return pd.DataFrame(filtered_rows)

    def _build_payload_per_qa(self, df_top5, query: str) -> str:
        results = []
        for _, row in df_top5.iterrows():
            conv = row["Conversation_History"]
            if isinstance(conv, str):
                try:
                    conv_json = json.loads(conv)
                    conversation = conv_json
                except Exception:
                    conversation = self._parse_conversation(conv)
            else:
                conversation = conv
            try:
                intents = json.loads(row["Entities"])
            except Exception:
                intents = {}
            try:
                relationships = json.loads(row["Relationships"])
            except Exception:
                relationships = []
            results.append({
                "company_name": row["Company_name"],
                "conversation": conversation,
                "intents": intents,
                "relationships": relationships
            })
        full_payload = {
            "query": query.strip(),
            "retrieved_answers": results
        }
        return json.dumps(full_payload, ensure_ascii=False, indent=2)

    def _query_llm(self, payload: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ENDBOT_PROMPT},
                {"role": "user", "content": payload}
            ],
            temperature=0,
            top_p=0.95
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_conversation(text: str) -> List[Dict[str, str]]:
        lines = text.split("\n")
        parsed = []
        for line in lines:
            lower = line.lower()
            if lower.startswith("customer"):
                role = "Customer"
                msg = line[len("Customer"):].strip()
            elif lower.startswith("company"):
                role = "Company"
                msg = line[len("Company"):].strip()
            else:
                role = "Customer" if not parsed else parsed[-1]["role"]
                msg = line.strip()
            if msg:
                parsed.append({"role": role, "message": msg})
        return parsed

# Example usage (remove or comment out in production if you want as a module):
if __name__ == "__main__":
    pipeline = ChatQAPipeline()
    input_text = "My Echo keeps playing the same song over and over, how do I fix recommendations?"

    # Both result and payload:
    response, rag_payload = pipeline.run_with_payload(input_text)
    print("LLM Response:\n", response)
    print("\nRAG Payload:\n", rag_payload)

    # Or separately:
    # print(pipeline.run(input_text))
    # print(pipeline.get_rag_payload(input_text))
