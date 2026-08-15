# 📄 RAG-Based Document Q&A

A Retrieval-Augmented Generation (RAG) application that allows users to upload **PDF or TXT documents** and ask questions about their content.

The system uses **Sentence Transformers** to generate embeddings, **FAISS** for semantic similarity search, and **Groq's Llama 3.1 8B Instant** model to generate answers grounded in the retrieved document context.

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://rag-based-document-qa-api.streamlit.app/

---

## ✨ Features

- 📄 Upload PDF and TXT documents
- ✂️ Overlapping text chunking
- 🧠 Semantic embeddings using Sentence Transformers
- 🔎 FAISS-based vector similarity search
- 🎯 Top-k relevant chunk retrieval
- 🤖 LLM-powered answer generation using Groq
- 📚 Retrieved source chunks with similarity scores
- 🌐 Interactive Streamlit interface
- ⚡ FastAPI REST API
- 📖 Automatic Swagger/OpenAPI documentation

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Backend | FastAPI, Uvicorn |
| RAG | Custom Python RAG Pipeline |
| Embeddings | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| LLM | Groq API |
| PDF Processing | pypdf |
| Frontend | Streamlit |
| API Testing | Swagger / OpenAPI |
| Deployment | Streamlit Community Cloud |
---

## 🏗️ Architecture

```text
                    PDF / TXT
                       │
                       ▼
                Streamlit UI
                       │
                       ▼
                Text Extraction
                       │
                       ▼
                 Text Chunking
                + Overlapping
                       │
                       ▼
          Sentence Transformer Model
              all-MiniLM-L6-v2
                       │
                       ▼
                  Embeddings
                       │
                       ▼
                     FAISS
              Vector Similarity Search
                       │
                       ▼
                Top-K Chunks
                       │
                       ▼
                Prompt Creation
                       │
                       ▼
                  Groq LLM
              Llama 3.1 8B Instant
                       │
                       ▼
              Grounded Answer
                       │
                       ▼
              Answer + Sources
```

---

## 🧠 How the RAG Pipeline Works

### 1. Document Ingestion

The user uploads a PDF or TXT file through the Streamlit interface.

For PDFs, text is extracted using `pypdf`.

### 2. Text Chunking

The extracted document is divided into smaller overlapping chunks.

The current implementation uses configurable chunk size and overlap. Keep these values synchronized with `app/rag.py`.

### 3. Embedding Generation

Each document chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

The embeddings are normalized before being stored in FAISS.

### 4. Vector Search

FAISS is used for efficient similarity search.

The project uses:

```python
faiss.IndexFlatIP
```

with normalized vectors, making inner-product search equivalent to cosine similarity.

### 5. Query Retrieval

When a user asks a question:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Similarity Search
   ↓
Top-K Relevant Chunks
```

The most relevant chunks are selected as context for the LLM.

### 6. Answer Generation

The retrieved chunks and the user's question are combined into a prompt.

The prompt is sent to the Groq API using:

```text
Llama 3.1 8B Instant
```

The model is instructed to answer using the retrieved document context and avoid unsupported information.

### 7. Sources

The application displays the retrieved chunks along with their similarity scores, allowing the user to inspect the context used to generate the answer.

---

## 📁 Project Structure

```text
RAG-based-Document-Q-A-API/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── rag.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

### `app/rag.py`

Contains the core RAG pipeline:

- Text chunking
- Embedding generation
- FAISS vector storage
- Similarity search
- Prompt construction
- Groq LLM integration
- Answer and source generation

### `app/main.py`

Contains the FastAPI backend and REST API endpoints.

### `streamlit_app.py`

Provides the user-facing interface for:

- Uploading PDF/TXT files
- Asking questions
- Displaying generated answers
- Displaying retrieved sources

---

## 🔌 FastAPI API

The project also exposes the RAG functionality through FastAPI.

Run:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

```text
GET  /health
POST /upload
POST /upload-file
POST /query
```

The FastAPI interface can be used to test the RAG pipeline independently of the Streamlit frontend.

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sam-3115/RAG-based-Document-Q-A-API.git
cd RAG-based-Document-Q-A-API
```

### 2. Create a virtual environment

For Python 3.11:

```bash
py -3.11 -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Windows Command Prompt:

```cmd
set "GROQ_API_KEY=YOUR_GROQ_API_KEY"
```

PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### 5. Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🔐 Environment Variables

The application requires:

```text
GROQ_API_KEY
```

For Streamlit Community Cloud, the API key is stored using **Streamlit Secrets** rather than being included in the repository.

---

### RAG Process

```text
Question
   ↓
Embedding
   ↓
FAISS Search
   ↓
Relevant Document Chunks
   ↓
Groq LLM
   ↓
Answer
```

### Output

The application provides:

- Generated answer
- Retrieved source chunks
- Similarity scores

---

## ⚠️ Limitations

### In-memory document storage

The current implementation stores document vector stores in memory.

Uploaded documents are lost when the application restarts.

A production system could use a persistent vector database.

### Fixed-size chunking

The current implementation uses character-based chunking with overlap.

More advanced implementations could use:

- Sentence-based chunking
- Recursive chunking
- Semantic chunking
- Document-aware chunking

### No reranking

The current pipeline directly uses FAISS similarity scores for retrieval.

A reranking model could be added to improve retrieval precision.

### Scanned PDFs

The current PDF extraction relies on `pypdf`, so image-only/scanned PDFs may require OCR before they can be processed effectively.

### Public deployment

The deployed demo uses a server-side Groq API key. The key is stored as a secret and is not exposed in the frontend.


The goal is to combine **semantic retrieval** with **LLM generation** so that answers are based on information retrieved from the user's document.

---
