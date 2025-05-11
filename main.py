from Py_files.reranker import CrossEncoderReranker
# This script demonstrates how to use the CrossEncoderReranker class to rerank a list of outputs based on a given query.
reranker = CrossEncoderReranker()

query = "Explain the benefits of machine learning"
rag_outputs = [
    "Machine learning enables systems to learn from data.",
    "Quantum mechanics describes subatomic particles.",
    "It allows automation of tasks without hard-coding rules.",
    "Deep learning is a subset of machine learning.",
    "Rain is caused by condensation of atmospheric moisture."
]

top5 = reranker.rerank(query, rag_outputs)

for idx, (text, score) in enumerate(top5, 1):
    print(f"{idx}. {score:.4f} - {text}")

