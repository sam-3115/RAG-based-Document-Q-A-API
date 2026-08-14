# RAG Document Q&A API

A small but real Retrieval-Augmented Generation (RAG) service: upload a document,
then ask questions about it and get answers grounded in the actual text — not
just whatever the LLM remembers or hallucinates.

## Architecture

```
Upload:  document text --> chunk_text() --> embed (sentence-transformers)
                                          --> store in FAISS index (per doc_id)

Query:   question --> embed --> cosine similarity search in FAISS
                    --> top-k relevant chunks
                    --> build prompt (question + retrieved context)
                    --> call LLM (Groq API)
                    --> grounded answer + source chunks returned
```

## Why these choices

- **Chunking**: sliding window with overlap so ideas that span a chunk
  boundary aren't lost. Chunk size and overlap are tunable constants.
- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers` — small,
  fast, free, runs locally (no API cost for the retrieval side).
- **Vector search**: FAISS `IndexFlatIP` on L2-normalized vectors, which
  is equivalent to cosine similarity — exact (not approximate) search,
  fine for small/medium document collections.
- **Generation**: Groq's free-tier API (OpenAI-compatible), swappable for
  Azure OpenAI, Gemini, or any other provider by changing `call_llm()` in
  `app/rag.py`.
- **FastAPI**: async-ready, automatic OpenAPI/Swagger docs at `/docs`,
  request/response validation via Pydantic models.

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Groq API key: https://console.groq.com
export GROQ_API_KEY="your_key_here"   # Windows: set GROQ_API_KEY=your_key_here

# 4. Run the server
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger UI — you can
try `/upload` and `/query` directly from the browser.

## Example usage (curl)

```bash
# Upload a document
curl -X POST http://127.0.0.1:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern Python web framework built on Starlette and Pydantic."}'
# -> {"doc_id": "...", "num_chunks": 1}

# Ask a question about it
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "PASTE_DOC_ID_HERE", "question": "What is FastAPI built on?"}'
# -> {"answer": "...", "sources": [{"text": "...", "score": 0.87}]}
```

## Known limitations (be upfront about these — they're good interview talking points)

- In-memory storage only — restarting the server loses all uploaded documents.
  A real system would use a persistent vector DB (Chroma, Pinecone, pgvector, etc.).
- No re-ranking step after retrieval (a cross-encoder re-ranker would improve
  precision on larger document sets).
- `/upload-file` only supports `.txt` — PDF support would need a parser like
  `pypdf` or `pdfplumber` added to the ingestion step.
- No authentication/rate-limiting — fine for a demo, not for production.
- Fixed chunk size/overlap rather than semantic chunking.

## Project structure

```
rag_fastapi_project/
├── app/
│   ├── __init__.py
│   ├── main.py       # FastAPI routes
│   └── rag.py         # chunking, embeddings, FAISS store, LLM call
├── requirements.txt
└── README.md
```
