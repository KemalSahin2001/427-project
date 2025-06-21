import json
from typing import List, Dict, Optional

def build_gpt_input_payload(
    query: str,
    rag_results: List[Dict],  # Each RAG result = {"company_name": ..., "conversation": [...]}
    intents: Optional[Dict] = None,  # e.g., {"products": [...], "services": [...], ...}
    relationships: Optional[List[Dict]] = None  # e.g., [{"subject": ..., "predicate": ..., "object": ...}]
) -> str:
    """
    Build input payload for GPT-4o-mini one-shot inference.
    
    Args:
        query: User question string.
        rag_results: List of up to 5 RAG results, each in the QA format.
        intents: Optional dictionary of extracted intents.
        relationships: Optional list of extracted relationships.

    Returns:
        A formatted JSON string to use as user message input in OpenAI API.
    """
    payload = {
        "query": query.strip(),
        "retrieved_answers": rag_results[:5]  # limit to top-5
    }

    if intents:
        payload["intents"] = intents

    if relationships:
        payload["relationships"] = relationships

    return json.dumps(payload, ensure_ascii=False, indent=2)


"""
query = "Why is no one responding to my private messages?"

rag_results = [
    {
        "company_name": "sprintcare",
        "conversation": [
            {"role": "Customer", "message": "is the worst customer service"},
            {"role": "Company", "message": "Can you please send us a private message so that I can gain further details about your account"},
            {"role": "Customer", "message": "I did"},
            {"role": "Company", "message": "Please send us a Private Message so that we can further assist you Just click Message at the top of your profile"},
            {"role": "Customer", "message": "I have sent several private messages and no one is responding as usual"},
            {"role": "Company", "message": "I understand I would like to assist you We would need to get you into a private secured link to further assist"}
        ]
    }
]

intents = {
    "products": ["OCS Account Takeover", "Consent Form"],
    "services": ["OCS Account Takeover"],
    "issue_types": ["incorrect information", "broken link"]
}

relationships = [
    {"subject": "poor customer service", "predicate": "resolvesWith", "object": "use secure link"},
    {"subject": "lack of response", "predicate": "resolvesWith", "object": "use secure link"}
]

text_input = build_gpt_input_payload(query, rag_results, intents, relationships)

# Then pass `text_input` to OpenAI API as `{"role": "user", "content": text_input}`


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": text}
    ],
    temperature=0,
    top_p=0.95
)
response.choices[0].message.content
"""