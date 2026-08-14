"""
FastAPI service for a RAG-based Document Q&A API.

Endpoints:
  POST /upload  -> accepts raw text (or a .txt/.pdf file), builds a vector index, returns doc_id
  POST /query   -> accepts a doc_id + question, returns a grounded answer + source chunks
  GET  /health  -> simple health check

Run locally:
  uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app import rag

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload a document, then ask grounded questions about it using retrieval-augmented generation.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class UploadTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to ingest")


class UploadResponse(BaseModel):
    doc_id: str
    num_chunks: int


class QueryRequest(BaseModel):
    doc_id: str = Field(..., description="doc_id returned by /upload")
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload_text(payload: UploadTextRequest):
    """Ingest raw text directly (useful for testing without a file)."""
    doc_id = rag.ingest_document(payload.text)
    store = rag.get_store(doc_id)
    return UploadResponse(doc_id=doc_id, num_chunks=len(store.chunks))


@app.post("/upload-file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Ingest a .txt file upload. (Extend this to parse PDFs with pypdf if needed.)"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported in this demo.")

    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="ignore")

    doc_id = rag.ingest_document(text)
    store = rag.get_store(doc_id)
    return UploadResponse(doc_id=doc_id, num_chunks=len(store.chunks))


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    try:
        result = rag.answer_question(payload.doc_id, payload.question, k=payload.top_k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )
