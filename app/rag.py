"""
Core RAG (Retrieval-Augmented Generation) pipeline.

Flow:
  Document -> chunk_text() -> embed_chunks() -> stored in FAISS index
  Question -> embed -> similarity search against FAISS -> top-k chunks
            -> build prompt with retrieved context -> call LLM -> answer
"""

import os
import re
import uuid
from dataclasses import dataclass, field

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 80        # characters of overlap between consecutive chunks
TOP_K = 3                 # number of chunks to retrieve per query

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on whitespace boundaries.

    A sliding window keeps `overlap` characters of context between chunks
    so an idea that spans a boundary isn't lost to either chunk alone.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # try not to cut mid-word: extend to the next space if we're not at the end
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap if end - overlap > start else end

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
@dataclass
class VectorStore:
    """A tiny in-memory FAISS-backed vector store, one index per uploaded document."""

    model: SentenceTransformer
    dim: int
    index: faiss.Index = field(init=False)
    chunks: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.index = faiss.IndexFlatIP(self.dim)  # inner product == cosine sim on normalized vectors

    def add(self, chunks: list[str]) -> None:
        if not chunks:
            return
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        self.index.add(np.array(embeddings, dtype="float32"))
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vec, dtype="float32"), min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


# ---------------------------------------------------------------------------
# In-memory "document DB": doc_id -> VectorStore
# In a real system this would be a persistent vector DB (e.g. a managed
# FAISS/Chroma instance or a cloud vector service) instead of a process-local dict.
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_stores: dict[str, VectorStore] = {}


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def ingest_document(text: str) -> str:
    """Chunk + embed a document, store it, and return a doc_id to query later."""
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    store = VectorStore(model=model, dim=dim)

    chunks = chunk_text(text)
    store.add(chunks)

    doc_id = str(uuid.uuid4())
    _stores[doc_id] = store
    return doc_id


def get_store(doc_id: str) -> VectorStore | None:
    return _stores.get(doc_id)


# ---------------------------------------------------------------------------
# LLM call (generation step)
# ---------------------------------------------------------------------------
def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer isn't in the context, say you don't know instead of "
        "guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def call_llm(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com and set it before calling /query."
        )

    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def answer_question(doc_id: str, question: str, k: int = TOP_K) -> dict:
    store = get_store(doc_id)
    if store is None:
        raise KeyError(f"No document found for doc_id={doc_id}")

    retrieved = store.search(question, k=k)
    context_chunks = [c for c, _score in retrieved]

    if not context_chunks:
        return {"answer": "No relevant context found in the document.", "sources": []}

    prompt = build_prompt(question, context_chunks)
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "sources": [{"text": c, "score": round(s, 4)} for c, s in retrieved],
    }
