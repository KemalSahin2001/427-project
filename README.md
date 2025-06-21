# Customer Support Chatbot with Retrieval-Augmented Generation (RAG)

This project implements an AI-powered customer support assistant using a **Retrieval-Augmented Generation (RAG)** architecture.  
By combining **semantic search** with **advanced generative models**, this system can accurately and efficiently automate customer support queries.  

The solution is designed for scalability — starting with the **aviation industry**, but adaptable to other domains and use cases.

---

## 📌 Project Objective

- Automate customer support to reduce human workload and operational costs.
- Improve accuracy, relevance, and response time for customer queries.
- Enable scalable multi-domain deployment (starting from aviation industry).
- Build an advanced RAG pipeline with modular components and evaluation tools.

---

## 🚀 Key Features

✅ End-to-end **RAG pipeline** for customer support  
✅ **Intent extraction**: products, services, issue types, relationships  
✅ **Vector search** with Elasticsearch  
✅ Advanced **reranking** of retrieved results  
✅ Streamlit-based **demo UI**  
✅ Modular, well-documented Python codebase  
✅ **RAGAS**-based evaluation pipeline  

---

## 🗂️ Project Structure

```
MAIN
├── data/                       # Datasets (raw, processed, embeddings, results)
│   ├── processed/
│   ├── raw/
│   ├── test_results/
├── Notebooks/                  # Jupyter notebooks (experiments, tryouts)
│   ├── Gpt Pipeline/
│   ├── Llama Tryouts/
├── py_files/                   # Core pipeline and helper scripts
│   ├── eval/                   # Evaluation pipeline (RAGAS)
│   ├── llm_pipeline/           # LLM extraction pipeline
│   ├── VectorDBStructure/      # Vector DB helpers
│   ├── AIAsistantPipeline.py   # Main RAG pipeline (production)
│   ├── QA_Pipeline.py          # QA pipeline
│   ├── CONFIG.py               # Config file (paths, prompts)
├── Reports/                    # Project reports and PDFs
├── streamlit_demo.py           # Streamlit app (demo UI)
├── examplepipeline.ipynb       # Example pipeline run
├── ragas_pipeline.ipynb        # RAGAS evaluation notebook
├── README.md                   # Project README (this file)
├── .env                        # Env variables (API keys, Elastic settings)
├── LICENSE
```

---

## 🛠️ Solution Workflow

### 1️⃣ Data Collection & Preprocessing

- Data source: [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Preprocessing steps:
    - Clean usernames, URLs, special characters
    - Normalize text
    - Tokenize, lemmatize
    - Structure into JSON format with:
        - `chat_id`, `company_name`, `conversation_history`, `entities`, `relationships`, `embedding`

---

### 2️⃣ Intent Extraction Pipeline

**Sequential extraction:**

1. **Product extraction**
2. **Service extraction**
3. **Issue type extraction**
4. **Relationship extraction**

→ Stored in structured format for downstream embedding & search.

---

### 3️⃣ Embedding & Vector Storage

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding generation**: 
    - Intent embedding
    - Full conversation embedding
- **Vector DB**: Elasticsearch

Search & storage:
- Embeddings pushed to Elastic index
- Cosine similarity for fast retrieval
- Enables "semantic search" of past conversations

---

### 4️⃣ Query Processing Pipeline

1. **Preprocessing & vectorization**
2. **Elastic Top-50 retrieval**
3. **Reranking** using cross-encoder (query + candidates)
4. **Top-5 selection**
5. **Response generation** with GPT-4o mini

---

### 5️⃣ Evaluation Pipeline

- Custom **RAGAS** score (Retrieval-Augmented Generation Accuracy Score)
- Combines:
    - Retrieval accuracy
    - Generation quality
    - Context alignment

**Result:**  
→ Achieved **RAGAS score of 74** on aviation dataset (100 queries).

---

## ▶️ How to Run

### 1️⃣ Setup environment

```bash
# Clone repo
git clone <your-repo-url>
cd <your-repo-folder>

# Create virtualenv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure .env
cp .env.example .env
# Add your OpenAI API key + Elasticsearch config
```

---

### 2️⃣ Run pipeline

```bash
# Run full pipeline
python py_files/AIAsistantPipeline.py
```

---

### 3️⃣ Run Streamlit demo UI

```bash
streamlit run streamlit_demo.py
```

---

## 📋 Example Usage

```python
from AIAsistantPipeline import ChatQAPipeline

pipeline = ChatQAPipeline()

response, rag_payload = pipeline.run_with_payload(
    "My Echo keeps playing the same song over and over, how do I fix recommendations?"
)

print("LLM Response:", response)
print("RAG Payload:", rag_payload)
```

---

## ⚙️ Dependencies

- Python 3.9+
- openai
- sentence-transformers
- elasticsearch
- streamlit
- pandas
- numpy
- tqdm
- python-dotenv

---

## 📊 Results

- Dataset: 100 test queries (aviation sector)
- RAGAS score: **74**
- High relevance and accuracy in generated responses
- Fast & scalable for real-time customer support

---

## 🧭 Roadmap

✅ Expand to new industries  
✅ Add voice command support  
✅ Optimize reranker for speed  
✅ Productionize inference pipeline  

---

## 👥 Contributors

- Kemal Şahin 
- Burak Kurt  

---

## 📄 License

This project is licensed under the MIT License.

---

