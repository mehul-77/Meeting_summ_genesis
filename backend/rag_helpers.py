# backend/rag_helpers.py
from typing import List, Tuple, Optional
import math
import requests
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from chromadb.api import API
import os
import time

# Chroma client config: local persisted storage (sqlite) is enabled by default.
# For simple in-memory only, you can set persist_directory=None.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

# Use Chroma's OpenAI-compatible embedding adapter? We'll wrap Cohere embeddings manually.
# We'll create a Chroma client and collection per request (or reuse a singleton in main.py).

# Cohere embedding model name (server-side)
COHERE_EMBED_MODEL = "embed-english-v2.0"  # Cohere's recommended model as of 2024-2025; change if needed

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[Tuple[str, int, int]]:
    """
    Split `text` into overlapping character chunks.
    Returns list of (chunk_text, start_char, end_char).
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

def cohere_embed_texts(api_key: str, texts: List[str]) -> List[List[float]]:
    """
    Call Cohere embeddings API (HTTP) to embed a list of texts.
    Returns list of embedding vectors.
    """
    if not api_key:
        raise ValueError("Missing Cohere API key for embeddings")
    url = "https://api.cohere.ai/v1/embed"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Cohere supports batching multiple inputs.
    payload = {
        "model": COHERE_EMBED_MODEL,
        "input": texts
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Cohere embed API failed: {resp.status_code} {resp.text}")
    j = resp.json()
    # Cohere returns 'embeddings' array
    embeddings = j.get("embeddings") or j.get("data") or None
    if embeddings is None:
        # older shapes
        raise RuntimeError(f"Unexpected Cohere embed response: {j}")
    return embeddings

def build_chroma_collection(collection_name: str = "meetings", persist: bool = True, client: Optional[API]=None):
    """
    Create or get a Chroma collection. Returns (client, collection).
    If a client is passed, reuse it.
    """
    # If a client is not provided, create one
    if client is None:
        # store on local disk; ensures persistence across restarts if persist_directory set
        chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
    else:
        chroma_client = client

    # get or create collection
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
    return chroma_client, collection

def upsert_embeddings_to_chroma(collection, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas=None):
    """
    Upsert vectors into chroma collection.
    """
    if metadatas is None:
        metadatas = [{} for _ in texts]
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

def query_chroma(collection, query_embedding: List[float], top_k: int = 4):
    """
    Query chroma collection and return the top_k results as (ids, distances, documents, metadatas)
    """
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["distances","documents","metadatas","ids"])
    # results dict contains lists of results per query
    if not results:
        return []
    # results fields are list-of-list (single query => index 0)
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    out = []
    for i, _id in enumerate(ids):
        out.append({
            "id": _id,
            "distance": distances[i] if i < len(distances) else None,
            "document": documents[i] if i < len(documents) else None,
            "metadata": metadatas[i] if i < len(metadatas) else None
        })
    return out
