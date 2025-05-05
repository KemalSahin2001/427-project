import re
import ast
import json
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError


def parse_string(s):
    """
    - Strips numpy array wrapper
    - Safely evaluates the literal
    - Parses nested JSON strings in Entities & Relationships
    """
    s_clean = re.sub(
        r"array\((?P<x>\[.*?\]),\s*dtype=[^)]+\)",
        r"\1",
        s,
        flags=re.S
    )
    data = ast.literal_eval(s_clean)

    for k in ("Entities", "Relationships"):
        if isinstance(data.get(k), str):
            try:
                data[k] = json.loads(data[k])
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {k}: {data[k]}")
    return data


def main():
    # Read Excel (raw strings in first column)
    df = pd.read_excel("TWCS4000Embedding.xlsx")

    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    index = "chat_embeddings"

    # Determine embedding dimension
    first_doc = parse_string(df.iloc[0, 0])
    dims = len(first_doc["Embedding"])

    # Delete and recreate index with object mapping for Entities & Relationships
    if es.indices.exists(index=index):
        es.indices.delete(index=index)

    mapping = {
        "mappings": {
            "properties": {
                "ChatID":               {"type": "keyword"},
                "Company_name":         {"type": "text"},
                "Conversation_History": {"type": "object"},
                "Entities":             {"type": "object"},
                "Relationships":        {"type": "object"},
                "Embedding": {
                    "type":       "dense_vector",
                    "dims":       dims,
                    "index":      True,
                    "similarity": "cosine"
                }
            }
        }
    }
    es.indices.create(index=index, body=mapping)

    # Prepare bulk actions
    actions = []
    for raw in df.iloc[:, 0]:
        try:
            doc = parse_string(raw)
        except Exception as e:
            print("Parse error, skipping row:", e)
            continue
        actions.append({"_index": index, "_source": doc})

    # Bulk index documents
    try:
        helpers.bulk(es, actions)
        print("Bulk indexing completed successfully.")
    except BulkIndexError as e:
        print("Bulk index failures:", e.errors)


if __name__ == "__main__":
    main()
