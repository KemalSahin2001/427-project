"""
QA Pipeline
===========
A self‑contained class that converts an end‑to‑end retrieval‑augmented
question‑answering workflow into a reusable Python module.

▪ Heavy components (vector DB, cross‑encoder reranker, OpenAI client) are
  constructed **once** in ``__init__`` and then re‑used for every incoming
  query.
▪ ``run_with_payload`` returns both the assistant answer and the full RAG
  payload so that downstream apps (e.g. Streamlit) can decide what to
  display.

Usage
-----
>>> from qa_pipeline import QAPipeline
>>> pipeline = QAPipeline()
>>> answer, payload = pipeline.run_with_payload("I accidentally booked …")
>>> print(answer)
>>> print(payload)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import openai
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# ─── Internal imports ──────────────────────────────────────────────────────── #
from .llm_pipeline import twcs_processor as processor
from .llm_pipeline.llm_extractor import LLMExtractor
from .llm_pipeline.reranker import CrossEncoderReranker
from .VectorDBStructure.db_structure import DatabaseStructure
from .VectorDBStructure.query import query_similar
from CONFIG import ENDBOT_PROMPT

class QAPipeline:
    """Reusable Retrieval‑Augmented Generation (RAG) pipeline."""

    def __init__(
        self,
        es_top_k: int = 50,
        rerank_top_k: int = 50,
        hybrid_weights: Tuple[float, float] = (0.7, 0.3),
        openai_api_key: str | None = None,
    ) -> None:
        """Create a pipeline instance.

        Parameters
        ----------
        es_top_k
            Number of candidates to retrieve from ElasticSearch.
        rerank_top_k
            Number of candidates to feed into the cross‑encoder.
        hybrid_weights
            Tuple of weights ``(elastic_w, rerank_w)`` for hybrid scoring.
        openai_api_key
            If *None*, the key is read from the ``OPENAI_API_KEY`` env‑var.
        """
        # Load secrets just once
        load_dotenv()
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        self.client = openai.OpenAI(api_key=key)

        # Heavy components (constructed once)
        self.db = DatabaseStructure()
        self.reranker = CrossEncoderReranker(top_k=rerank_top_k)

        # Config
        self.es_top_k = es_top_k
        self.rerank_top_k = rerank_top_k
        self.w_sim, self.w_rerank = hybrid_weights

    # ── Public API ────────────────────────────────────────────────────────── #

    def run_with_payload(self, query: str, top_n: int = 5) -> Tuple[str, str]:
        """Answer *query* and return (assistant_answer, rag_payload_json)."""
        # 1) Pre‑process user query
        cleaned_conv, structured_conv = self._preprocess_query(query)

        # 2) Extract entities & relationships via LLM
        entities, relationships = self._extract_intents(structured_conv)

        # 3) Embed query for vector search
        embedding = self.db.text_to_embedding(
            cleaned_conv, entities, relationships
        ).tolist()

        # 4) Retrieve + cross‑encoder‑rerank
        hits = query_similar(embedding, k=self.es_top_k)
        candidates = [h["_source"]["Conversation_History"]["conversation"] for h in hits]
        reranked = self.reranker.rerank(query, candidates)

        # 5) Merge scores & compute hybrid ranking
        hybrid_df = self._build_hybrid_dataframe(query, hits, reranked)
        topk_df = self._select_diverse_topk(hybrid_df, k=top_n)

        # 6) Build RAG payload & call LLM for final answer
        payload = self._build_payload(topk_df, query)
        answer = self._call_llm(payload)
        return answer, payload

    # ── Internal helpers ──────────────────────────────────────────────────── #

    @staticmethod
    def _clean_single(text: str) -> str:
        return processor.TWCSProcessor._clean_single(text)

    @staticmethod
    def _structurize(text: str) -> str:
        return processor.TWCSProcessor._convert_to_conversation(text)

    def _preprocess_query(self, query: str) -> Tuple[str, str]:
        cleaned = self._clean_single(query)
        structured = self._structurize(cleaned)
        return cleaned, structured

    def _extract_intents(self, structured_conv: str) -> Tuple[dict, list]:
        df = pd.DataFrame([[structured_conv, structured_conv]],
                          columns=["cleaned_conversation", "structured_conversations"])
        pipe = LLMExtractor(dataframe=df)
        df1 = pipe.extract_entities()
        df2 = pipe.process_entities_json()
        df3 = pipe.extract_relationships()
        entities = df3["entities"].values[0]
        relationships = self.db.fix_relationships(df3["relationship"].values[0])
        return entities, relationships

    def _build_hybrid_dataframe(
        self,
        query: str,
        hits: List[dict],
        reranked: List[Tuple[str, float]],
    ) -> pd.DataFrame:
        # map conversation → (score, rank)
        score_rank = {c: (s, r + 1) for r, (c, s) in enumerate(reranked)}
        rows: List[dict] = []
        for h in hits:
            src = h["_source"]
            conv = src["Conversation_History"]["conversation"]
            rerank_score, rerank_rank = score_rank.get(conv, (0.0, None))
            rows.append(
                {
                    "prompt": query,
                    "id": h["_id"],
                    "similarity_score": h["_score"],
                    "rerank_score": rerank_score,
                    "rerank_rank": rerank_rank,
                    "ChatID": src["ChatID"],
                    "Company_name": src["Company_name"],
                    "Conversation_History": conv,
                    "Entities": json.dumps(src["Entities"]),
                    "Relationships": json.dumps(src["Relationships"]),
                }
            )
        df = pd.DataFrame(rows)
        scaler = StandardScaler()
        df[["sim_norm", "rerank_norm"]] = scaler.fit_transform(
            df[["similarity_score", "rerank_score"]].fillna(0)
        )
        df["hybrid_score"] = self.w_sim * df["sim_norm"] + self.w_rerank * df["rerank_norm"]
        return df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _select_diverse_topk(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        seen: set = set()
        picks: List[dict] = []
        for _, row in df.iterrows():
            key = (row["Entities"], row["Relationships"])
            if key not in seen:
                picks.append(row)
                seen.add(key)
            if len(picks) == k:
                break
        return pd.DataFrame(picks)

    # ── Payload & LLM call helpers ────────────────────────────────────────── #

    @staticmethod
    def _parse_conversation(text: str) -> List[Dict[str, str]]:
        parsed: List[Dict[str, str]] = []
        for line in text.split("\n"):
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

    def _build_payload(self, df: pd.DataFrame, query: str) -> str:
        results: List[dict] = []
        for _, row in df.iterrows():
            conv_raw = row["Conversation_History"]
            conv = (
                json.loads(conv_raw)
                if isinstance(conv_raw, str) and conv_raw.strip().startswith("[")
                else self._parse_conversation(conv_raw)
            )
            try:
                intents = json.loads(row["Entities"])
            except Exception:
                intents = {}
            try:
                relationships = json.loads(row["Relationships"])
            except Exception:
                relationships = []
            results.append(
                {
                    "company_name": row["Company_name"],
                    "conversation": conv,
                    "intents": intents,
                    "relationships": relationships,
                }
            )
        payload = {"query": query.strip(), "retrieved_answers": results}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _call_llm(self, payload: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ENDBOT_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0,
            top_p=0.95,
        )
        return response.choices[0].message.content


# ─── CLI quick‑test ────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    q = "I accidentally booked the same flight twice—VX666 and VX667. Please refund one."
    pipeline = QAPipeline()
    answer, rag = pipeline.run_with_payload(q)
    print("\n— Assistant —\n", answer)
    print("\n— RAG Payload (truncated) —\n", rag[:500], "…")
