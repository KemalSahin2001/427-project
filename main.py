from AIAsistantPipeline import ChatQAPipeline

pipeline = ChatQAPipeline()

response, rag_payload = pipeline.run_with_payload(
    "My Echo keeps playing the same song over and over, how do I fix recommendations?"
)
print("LLM cevabı:", response)
print("\nRAG Payload:\n", rag_payload)


