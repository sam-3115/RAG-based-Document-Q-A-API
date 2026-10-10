
"""
Core RAG pipeline:
Document -> chunks -> embeddings -> FAISS
Question -> retrieve chunks -> prompt -> Gemini answer
"""

import os
import re
import uuid
from dataclasses import dataclass, field

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3
GEMINI_MODEL = "gemini-2.5-flash"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1:
                end = start + last_space
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c]


@dataclass
class VectorStore:
    model: SentenceTransformer
    dim: int
    index: faiss.Index = field(init=False)
    chunks: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.index = faiss.IndexFlatIP(self.dim)

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
        scores, indices = self.index.search(
            np.array(query_vec, dtype="float32"),
            min(k, self.index.ntotal),
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


_embedding_model: SentenceTransformer | None = None
_stores: dict[str, VectorStore] = {}


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def ingest_document(text: str) -> str:
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


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context_chunks)
    )
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer isn't in the context, say you don't know instead of "
        "guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def call_llm(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )
    return llm.invoke(prompt).content


def answer_question(doc_id: str, question: str, k: int = TOP_K) -> dict:
    store = get_store(doc_id)
    if store is None:
        raise KeyError(f"No document found for doc_id={doc_id}")

    retrieved = store.search(question, k=k)
    context_chunks = [c for c, _score in retrieved]

    if not context_chunks:
        return {
            "answer": "No relevant context found in the document.",
            "sources": [],
        }

    answer = call_llm(build_prompt(question, context_chunks))
    return {
        "answer": answer,
        "sources": [
            {"text": c, "score": round(s, 4)}
            for c, s in retrieved
        ],
    }
