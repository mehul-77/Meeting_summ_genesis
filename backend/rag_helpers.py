# backend/rag_helpers.py
"""
RAG helper utilities.

This version:
- Uses chromadb.Client(...) (no 'API' import) when available.
- Falls back to a simple in-memory vector store using numpy if chromadb is not installed or fails.
- Provides:
    - chunk_text(...)
    - hf_embeddings(...)  -> calls Hugging Face embeddings endpoint
    - build_chroma_collection(...) -> returns (client, collection) OR (None, fallback_store)
    - upsert_embeddings_to_chroma(...)
    - query_chroma(...)
    - hf_generate(...) -> calls HF generation endpoint
"""
from typing import List, Tuple, Optional, Any
import os
import requests
import math
import logging

# try chromadb import; if unavailable, we'll use a fallback
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except Exception:
    chromadb = None
    Settings = None
    HAS_CHROMA = False

import numpy as np

# HF models (env override possible)
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "google/flan-t5-large")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping character chunks.
    Returns [(chunk_text, start_char, end_char), ...]
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def hf_embeddings(hf_api_key: str, texts: List[str], model: str = None) -> List[List[float]]:
    """
    Call Hugging Face Inference embeddings endpoint.
    Returns list of embedding vectors (list of floats).
    """
    if model is None:
        model = HF_EMBED_MODEL
    url = f"https://api-inference.huggingface.co/embeddings"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"model": model, "input": texts}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"HF embeddings failed: {resp.status_code} {resp.text}")
    j = resp.json()
    if isinstance(j, dict) and "embeddings" in j:
        return j["embeddings"]
    if isinstance(j, list):
        return j
    raise RuntimeError(f"Unexpected HF embeddings response: {j}")


# ---------- Chroma wrapper (or fallback) ----------

class InMemoryVectorStore:
    """Simple in-memory vector store fallback using numpy + cosine similarity."""
    def __init__(self):
        self.ids = []
        self.docs = []
        self.embeddings = []
        self.metadatas = []

    def upsert(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: Optional[List[dict]] = None):
        if metadatas is None:
            metadatas = [{} for _ in documents]
        for i, _id in enumerate(ids):
            self.ids.append(_id)
            self.docs.append(documents[i])
            self.embeddings.append(np.array(embeddings[i], dtype=np.float32))
            self.metadatas.append(metadatas[i])

    def query(self, query_embeddings: List[List[float]], n_results: int = 4, include=None):
        q = np.array(query_embeddings[0], dtype=np.float32)
        if len(self.embeddings) == 0:
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
        arr = np.vstack(self.embeddings)
        # cosine similarity
        norms = np.linalg.norm(arr, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        sims = (arr @ q) / norms
        # get topk indices by similarity
        topk = min(n_results, len(sims))
        idxs = np.argsort(-sims)[:topk]
        ids = [self.ids[i] for i in idxs]
        docs = [self.docs[i] for i in idxs]
        dists = [float(1.0 - sims[i]) for i in idxs]  # distance-like
        metas = [self.metadatas[i] for i in idxs]
        return {"ids": [ids], "documents": [docs], "distances": [dists], "metadatas": [metas]}


def build_chroma_collection(collection_name: str, persist: bool = True):
    """
    Returns (client, collection) if chromadb is available, otherwise (None, InMemoryVectorStore()).
    collection_name used only for chromadb.
    """
    if HAS_CHROMA:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)
        return client, collection
    else:
        # fallback in-memory
        logger.warning("chromadb not available — using in-memory fallback vector store.")
        store = InMemoryVectorStore()
        return None, store


def upsert_embeddings_to_chroma(collection: Any, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas=None):
    """
    Upsert embeddings into chroma collection or fallback store.
    """
    if hasattr(collection, "upsert"):
        # chromadb collection has upsert with named args
        try:
            collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas or [{} for _ in texts])
        except TypeError:
            # older chromadb signatures
            collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas or [{} for _ in texts])
    else:
        # fallback store
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


# Compatibility wrappers for older API (build_chroma, upsert_chroma)
def build_chroma(collection_name: str, persist: bool = True):
    """Compatibility wrapper for legacy name build_chroma.
    Returns the same as build_chroma_collection.
    """
    logger.warning("build_chroma is deprecated; use build_chroma_collection")
    return build_chroma_collection(collection_name, persist)


def upsert_chroma(collection: Any, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas=None):
    """Compatibility wrapper for legacy name upsert_chroma.
    Calls upsert_embeddings_to_chroma under the hood.
    """
    logger.warning("upsert_chroma is deprecated; use upsert_embeddings_to_chroma")
    return upsert_embeddings_to_chroma(collection, ids, texts, embeddings, metadatas)


def query_chroma(collection: Any, query_embedding: List[float], top_k: int = 4):
    """
    Query chroma collection or fallback store and return list of dicts:
      [{ "id":..., "document":..., "distance":..., "metadata":... }, ...]
    """
    # chroma returns dict-like with lists per query
    if HAS_CHROMA:
        res = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
        if not res:
            return []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out = []
        for i, _id in enumerate(ids):
            out.append({"id": _id, "document": docs[i] if i < len(docs) else None, "distance": dists[i] if i < len(dists) else None, "metadata": metas[i] if i < len(metas) else None})
        return out
    else:
        res = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents","metadatas","distances","ids"])
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out = []
        for i, _id in enumerate(ids):
            out.append({"id": _id, "document": docs[i] if i < len(docs) else None, "distance": dists[i] if i < len(dists) else None, "metadata": metas[i] if i < len(metas) else None})
        return out


def hf_generate(hf_api_key: str, prompt: str, model: str = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Call HF text generation model via Inference API and return the generated text.
    """
    if model is None:
        model = HF_GEN_MODEL
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_api_key}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature, "return_full_text": False}}
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"HF generate failed: {resp.status_code} {resp.text}")
    j = resp.json()
    # handle common response shapes
    if isinstance(j, dict) and "generated_text" in j:
        return j["generated_text"]
    if isinstance(j, list) and len(j) > 0:
        first = j[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
    if isinstance(j, str):
        return j
    return str(j)
