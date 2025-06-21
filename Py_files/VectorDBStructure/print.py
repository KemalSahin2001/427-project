from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200")

# scan all documents in the index
for hit in helpers.scan(
    client=es,
    index="chat_embeddings",
    query={"query": {"match_all": {}}},
    _source=["ChatID", "Company_name", "Embedding"]  # fetch only what you need
):
    src = hit["_source"]
    print(f"ChatID={src['ChatID']} Company={src['Company_name']}")
    # print("Embedding:", src["Embedding"])
    print("-" * 40)
