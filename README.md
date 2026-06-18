# 📚 RAG — Retrieval-Augmented Generation Pipeline

A end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload PDF documents, embed their content into a persistent vector store, and query them using an LLM — grounding every answer in your own documents.

---

## 🧠 How It Works

```
PDF Documents
     │
     ▼
Document Loader       ← Load & parse PDFs (LangChain)
     │
     ▼
Text Chunker          ← Split into overlapping chunks
     │
     ▼
Embedding Model       ← Convert chunks → dense vectors (SentenceTransformers)
     │
     ▼
Vector Store          ← Persist embeddings + metadata (ChromaDB)
     │
     ▼
Query Engine          ← Retrieve top-k chunks → LLM (Groq)
     │
     ▼
Answer
```

---

## 🗂️ Project Structure

```
RAG/
├── notebook/
│   └── pdf_loader.ipynb      # Main pipeline notebook
├── data/
│   └── vector_store/         # Persisted ChromaDB collection
├── .env                      # API keys (never committed)
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Muhammad-Musharraf/RAG.git
cd RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.  
> Get your Groq API key at [console.groq.com](https://console.groq.com).

---

## 🚀 Usage

Open and run the notebook step by step:

```bash
jupyter notebook notebook/pdf_loader.ipynb
```

### Pipeline steps inside the notebook

```python
# 1. Load PDFs
documents = load_pdfs("./data/pdfs/")

# 2. Chunk documents
chunked_documents = chunk_documents(documents)

# 3. Generate embeddings
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc.page_content for doc in chunked_documents]
embeddings = embedding_model.encode(texts, batch_size=32, convert_to_numpy=True)

# 4. Store in vector store
vector_store = VectorStore()
vector_store.add_documents(chunked_documents, embeddings)

# 5. Query
results = vector_store.query("What is the document about?")
```

---

## 🧩 Core Components

### `VectorStore`

A persistent document store backed by ChromaDB.

```python
vector_store = VectorStore(
    collection_name="pdf_documents",     # ChromaDB collection name
    persist_directory="./data/vector_store"  # Where embeddings are saved on disk
)
```

| Method | Description |
|--------|-------------|
| `initialize_store()` | Creates or loads the ChromaDB collection |
| `add_documents(documents, embeddings)` | Embeds and stores chunked documents with metadata |

### Embedding Model

Uses `sentence-transformers/all-MiniLM-L6-v2` — a lightweight model producing 384-dimensional embeddings, well-suited for semantic search.

### LLM

Uses **Groq** (fast inference) to generate answers grounded in retrieved document chunks.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `chromadb` | Persistent vector store |
| `sentence-transformers` | Text embedding model |
| `langchain` | Document loading & chunking |
| `groq` | LLM inference |
| `python-dotenv` | Load API keys from `.env` |
| `numpy` | Embedding array operations |
| `uuid` | Unique document IDs |

Install all at once:

```bash
pip install chromadb sentence-transformers langchain langchain-community \
            groq python-dotenv numpy
```

---

## 🔐 Security

- **API keys must be stored in `.env`** and loaded with `python-dotenv`
- `.env` is in `.gitignore` — never stage or commit it
- GitHub push protection will block any push containing secrets

```python
# Correct way to load keys
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
