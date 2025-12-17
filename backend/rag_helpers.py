# backend/rag_helpers.py
"""
RAG helper utilities with robust chroma fallback and name sanitization.

Exposes:
- chunk_text(...)
- hf_embeddings(...)
- build_chroma_collection(...)
- upsert_embeddings_to_chroma(...)
- query_chroma(...)
- hf_generate(...)
"""

from typing import List, Tuple, Optional, Any
import os
import re
import requests
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Try importing chromadb; if not present, use fallback option
try:
    import chromadb
    HAS_CHROMA = True
except Exception as e:
    chromadb = None
    HAS_CHROMA = False
    logger.info(f"chromadb not available or failed to import: {e}")

# Environment variables for configuration
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "google/flan-t5-large")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


def sanitize_collection_name(name: str) -> str:
    """Ensure collection names follow Chroma rules."""
    if not name:
        return "col-startup"
    # Replace invalid characters with hyphen
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9]+$", "", sanitized)

    if len(sanitized) < 3:
        sanitized = f"col-{sanitized}" if sanitized else "col-startup"

    return sanitized[:512]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[Tuple[str, int, int]]:
    """Split text into overlapping character chunks."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((text[start:end], start, end))
        start = end - overlap
    return chunks


def hf_embeddings(api_key: str, texts: List[str], model: str = HF_EMBED_MODEL) -> List:
    """Generate embeddings using Hugging Face API."""
    url = "https://router.huggingface.co/inference/embeddings"
    payload = {"model": model, "inputs": texts}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "embeddings" not in data:
            raise RuntimeError("HF response missing 'embeddings'")
        
        return data["embeddings"]

    except requests.RequestException as e:
        logger.exception(f"Failed to fetch embeddings: {e}")
        raise RuntimeError(f"HF embeddings error: {e}")


# ----------------- In-memory fallback vector store -----------------

class InMemoryVectorStore:
    """Simple fallback vector store using numpy for cosine similarity."""
    
    def __init__(self):
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.embs: List[np.ndarray] = []
        self.metadatas: List[dict] = []

    def upsert(self, ids, documents, embeddings, metadatas=None):
        metadatas = metadatas or [{} for _ in documents]

        for i, doc_id in enumerate(ids):
            self.ids.append(doc_id)
            self.docs.append(documents[i])
            self.embs.append(np.array(embeddings[i], dtype=np.float32))
            self.metadatas.append(metadatas[i])

    def query(self, query_embedding, n_results=4):
        if not self.embs:
            return {"ids": [], "documents": [], "distances": [], "metadatas": []}
        
        query_vector = np.array(query_embedding, dtype=np.float32)
        all_embeddings = np.vstack(self.embs)

        similarity = np.dot(all_embeddings, query_vector) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_vector)
        )

        top_indices = np.argsort(-similarity)[:n_results]
        return {
            "ids": [self.ids[i] for i in top_indices],
            "documents": [self.docs[i] for i in top_indices],
            "distances": [1 - similarity[i] for i in top_indices],
            "metadatas": [self.metadatas[i] for i in top_indices],
        }


# ----------------- Chroma Wrapper -----------------

def build_chroma_collection(collection_name: str, persist: bool = True):
    """Create or retrieve Chroma collection."""
    collection_name = sanitize_collection_name(collection_name)
    
    if not HAS_CHROMA:
        logger.info("chromadb not installed. Falling back to in-memory store.")
        return None, InMemoryVectorStore()

    try:
        client = chromadb.Client()
        collection = client.get_collection(name=collection_name)
        return client, collection
    except Exception as e:
        logger.warning(f"Chroma initialization failed: {e}")
        return None, InMemoryVectorStore()


def upsert_embeddings(collection, ids, texts, embeddings, metadatas=None):
    """Insert or update embeddings in the collection."""
    try:
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        logger.info("Successfully upserted embeddings.")
    except Exception as e:
        logger.exception(f"Error during upsert: {e}")
        raise RuntimeError("Failed to upsert embeddings")


def hf_generate(hf_api_key: str, prompt: str, model: Optional[str] = None) -> str:
    """Generate text completions with Hugging Face."""
    model = model or HF_GEN_MODEL
    url = "https://huggingface.co/api/generate"

    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"model": model, "input": prompt}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json().get("output", "")
    except requests.RequestException as e:
        logger.exception(f"Hugging Face generation failed: {e}")
        raise RuntimeError("Text generation failed")
