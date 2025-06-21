import pandas as pd
import ast
from datasets import Dataset
from ragas.metrics import answer_relevancy
from ragas import evaluate
from langchain_community.chat_models import ChatOpenAI
import os
from statistics import mean

# 🔑 Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your real key

# 📥 Load Excel file
df = pd.read_excel("results_cleaned.xlsx")


def parse_row(row):
    try:
        # Load list of retrievals (dicts)
        retrievals = ast.literal_eval(row["retrievals"])
        # Extract all company responses as text
        contexts = []
        for item in retrievals:
            for msg in item.get("conversation", []):
                if msg.get("role") == "Company":
                    contexts.append(msg["message"])
        return {
            "question": str(row["prompts"]),
            "answer": str(row["answers"]),
            "contexts": contexts  # ✅ list of strings
        }
    except Exception as e:
        print("❌ Failed to parse retrievals for row:\n", row["retrievals"])
        raise e


# 🛠 Build data for RAGAS
parsed_data = [parse_row(row) for _, row in df.iterrows()]

# 🧪 Optional: show one parsed sample
print("✅ First sample:\n", parsed_data[0])

# 📦 Convert to HuggingFace Dataset
dataset = Dataset.from_list(parsed_data)

# 🧠 Run RAGAS evaluation
results = evaluate(
    dataset,
    metrics=[answer_relevancy],
)

# 📊 Print average and per-sample scores
answer_scores = results["answer_relevancy"]

print("\n📈 Overall Answer Relevancy:", round(mean(answer_scores), 4))

print("\n📄 Per-sample scores:")
for i, score in enumerate(answer_scores):
    print(f"Sample {i + 1}: {score:.4f}")

# Custom eval is applied afterwards
