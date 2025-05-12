from elasticsearch import Elasticsearch


def query_similar(embedding, k=5, index="chat_embeddings", host="localhost", port=9200):
    # connect with scheme
    es = Elasticsearch(f"http://{host}:{port}", basic_auth=("elastic", "*pwASJfphV27RFS=BSWH"))

    # fetch embedding dim
    mapping = es.indices.get_mapping(index=index)
    dims = mapping[index]["mappings"]["properties"]["Embedding"]["dims"]
    if len(embedding) != dims:
        raise ValueError(
            f"Dimension mismatch: got {len(embedding)}, expected {dims}")
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    # To ensure non-negativity
                    "source": "cosineSimilarity(params.query_vector, 'Embedding') + 1.0",
                    "params": {"query_vector": embedding}
                }
            }
        }
    }
    return es.search(index=index, body=body)["hits"]["hits"]


def main():
    # zero‐vector of correct length (replace 384 if different)
    embedding = [1.0] * 384
    k = 5
    try:
        results = query_similar(embedding, k)
    except Exception as e:
        print("Error:", e)
        exit(1)

    for hit in results:
        src = hit["_source"]
        print(
            f"ID={hit['_id']} score={hit['_score']:.4f} ChatID={src.get('ChatID')}")


if __name__ == "__main__":
    main()
