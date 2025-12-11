# backend/rag_helpers.py
"""
RAG helper utilities with a robust chroma fallback.

This file:
- Attempts to use chromadb.Client() (modern client) without deprecated Settings().
- If chromadb import or client creation fails, falls back to a small in-memory vector store.
- Exposes:
    chunk_text(...)
    hf_embeddings(...)
    build_chroma_collection(...)
    upsert_embeddings_to_chroma(...)
    query_chroma(...)
    hf_generate(...)
"""

from typing import List, Tuple, Optional, Any
import os
import requests
import logging

logger = logging.getLogger(__name__)

# Try importing chromadb; if not present, we'll use fallback
try:
    import chromadb
    HAS_CHROMA = True
except Exception as e:
    chromadb = None
    HAS_CHROMA = False
    logger.info("chromadb not available or failed to import: %s", e)

import numpy as np

# Models (env override possible)
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "google/flan-t5-large")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[Tuple[str, int, int]]:
    """Split text into overlapping character chunks."""
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


def hf_embeddings(hf_api_key: str, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Call Hugging Face Inference embeddings endpoint.
    Returns list of vectors (lists of floats).
    """
    if model is None:
        model = HF_EMBED_MODEL
    url = "https://api-inference.huggingface.co/embeddings"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"model": model, "input": texts}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"HF embeddings failed: {resp.status_code} {resp.text}")
    j = resp.json()
    # HF may return dict with 'embeddings' or list directly
    if isinstance(j, dict) and "embeddings" in j:
        return j["embeddings"]
    if isinstance(j, list):
        return j
    raise RuntimeError(f"Unexpected HF embeddings response shape: {j}")


# ----------------- In-memory fallback vector store -----------------

class InMemoryVectorStore:
    """Simple fallback vector store using numpy and cosine similarity."""
    def __init__(self):
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.embs: List[np.ndarray] = []
        self.metadatas: List[dict] = []

    def upsert(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: Optional[List[dict]] = None):
        if metadatas is None:
            metadatas = [{} for _ in documents]
        for i, _id in enumerate(ids):
            self.ids.append(_id)
            self.docs.append(documents[i])
            self.embs.append(np.array(embeddings[i], dtype=np.float32))
            self.metadatas.append(metadatas[i])

    def query(self, query_embeddings: List[List[float]], n_results: int = 4, include=None):
        if len(self.embs) == 0:
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
        q = np.array(query_embeddings[0], dtype=np.float32)
        arr = np.vstack(self.embs)  # shape (N, D)
        # cosine similarity
        arr_norm = np.linalg.norm(arr, axis=1)
        q_norm = np.linalg.norm(q)
        denom = arr_norm * (q_norm if q_norm != 0 else 1.0)
        denom = np.where(denom == 0, 1e-10, denom)
        sims = (arr @ q) / denom
        topk = min(n_results, len(sims))
        idxs = np.argsort(-sims)[:topk]
        ids = [self.ids[i] for i in idxs]
        docs = [self.docs[i] for i in idxs]
        dists = [float(1.0 - sims[i]) for i in idxs]
        metas = [self.metadatas[i] for i in idxs]
        return {"ids": [ids], "documents": [docs], "distances": [dists], "metadatas": [metas]}


# ----------------- Chroma wrapper (new client usage) -----------------

def build_chroma_collection(collection_name: str, persist: bool = True):
    """
    Create or return a chroma collection (client, collection).
    If chromadb is not available or client init fails, returns (None, InMemoryVectorStore()).
    This avoids deprecated Settings(...) usage.
    """
    if not HAS_CHROMA:
        logger.info("Using in-memory vector store (chromadb not available).")
        return None, InMemoryVectorStore()

    # Try modern client creation without the legacy Settings(...) call.
    try:
        # Most chromadb installs support chromadb.Client() with optional Settings()
        # We prefer the simple constructor to avoid legacy config errors.
        client = None
        try:
            client = chromadb.Client()
        except TypeError:
            # Some older/newer versions may require Settings; fall back gently
            try:
                from chromadb.config import Settings  # try import locally
                client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIR))
            except Exception as e:
                logger.warning("chromadb.Client(Settings(...)) also failed: %s", e)
                client = None

        if client is None:
            logger.warning("Failed to construct chromadb client; using in-memory fallback.")
            return None, InMemoryVectorStore()

        # create or get collection
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)
        logger.info("Using chromadb collection '%s' (persist=%s).", collection_name, persist)
        return client, collection

    except Exception as e:
        logger.exception("chromadb initialization failed, falling back to in-memory store: %s", e)
        return None, InMemoryVectorStore()


def upsert_embeddings_to_chroma(collection: Any, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[dict]] = None):
    """
    Upsert to a chroma collection or in-memory fallback.
    """
    if hasattr(collection, "upsert"):
        # chromadb collections support upsert
        try:
            collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas or [{} for _ in texts])
            return
        except TypeError:
            # older api might use add(...)
            try:
                collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas or [{} for _ in texts])
                return
            except Exception as e:
                logger.exception("Chroma collection upsert/add failed: %s", e)
                raise
    # fallback store (our InMemoryVectorStore)
    if isinstance(collection, InMemoryVectorStore):
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        return
    # last resort: raise
    raise RuntimeError("Collection does not support upsert and is not fallback store.")


def query_chroma(collection: Any, query_embedding: List[float], top_k: int = 4):
    """
    Query chroma collection or fallback in-memory store.
    Returns list of dicts: [{id, document, distance, metadata}, ...]
    """
    if hasattr(collection, "query"):
        res = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
        if not res:
            return []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"id": ids[i], "document": docs[i] if i < len(docs) else None, "distance": dists[i] if i < len(dists) else None, "metadata": metas[i] if i < len(metas) else None} for i in range(len(ids))]
    else:
        # fallback InMemoryVectorStore's query returns same wrapper shape
        res = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"id": ids[i], "document": docs[i] if i < len(docs) else None, "distance": dists[i] if i < len(dists) else None, "metadata": metas[i] if i < len(metas) else None} for i in range(len(ids))]


def hf_generate(hf_api_key: str, prompt: str, model: Optional[str] = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Call HF generation endpoint and return generated text.
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
    # various response shapes handled
    if isinstance(j, dict) and "generated_text" in j:
        return j["generated_text"]
    if isinstance(j, list) and len(j) > 0:
        first = j[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
    if isinstance(j, str):
        return j
    return str(j)
