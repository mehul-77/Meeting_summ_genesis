# backend/rag_helpers.py
from typing import List, Tuple, Optional
import os
import requests
import chromadb
from chromadb.config import Settings
from chromadb.api import API

# Hugging Face model names (you may change)
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "google/flan-t5-large")

# Chroma persistence (local dir)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[Tuple[str,int,int]]:
    """
    Split text into overlapping character chunks.
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

def hf_embeddings(hf_api_key: str, texts: List[str], model: str = None) -> List[List[float]]:
    """
    Call HF Inference embeddings API. Returns list of vectors (lists of floats).
    """
    if model is None:
        model = HF_EMBED_MODEL
    url = f"https://api-inference.huggingface.co/embeddings"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    # The HF embeddings endpoint accepts JSON body with model and inputs
    payload = {
        "model": model,
        "inputs": texts
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code not in (200,201):
        raise RuntimeError(f"HF embeddings failed: {resp.status_code} {resp.text}")
    j = resp.json()
    # HF returns {'embeddings': [...] } or a list depending on infra; handle both
    if isinstance(j, dict) and "embeddings" in j:
        return j["embeddings"]
    # sometimes response is directly the list
    if isinstance(j, list):
        return j
    raise RuntimeError(f"Unexpected HF embeddings response: {j}")

def build_chroma(collection_name: str, persist: bool = True):
    """
    Create or get a Chroma client + collection.
    Returns (client, collection).
    """
    if persist:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
    else:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=None))
    try:
        col = client.get_collection(name=collection_name)
    except Exception:
        col = client.create_collection(name=collection_name)
    return client, col

def upsert_chroma(collection, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas=None):
    if metadatas is None:
        metadatas = [{} for _ in texts]
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

def query_chroma(collection, query_embedding: List[float], top_k: int = 4):
    res = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents","metadatas","distances","ids"])
    if not res:
        return []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for i,_id in enumerate(ids):
        out.append({"id": _id, "document": docs[i] if i < len(docs) else None, "distance": dists[i] if i < len(dists) else None, "metadata": metas[i] if i < len(metas) else None})
    return out

def hf_generate(hf_api_key: str, prompt: str, model: str = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Call HF text generation model via Inference API and return the generated text.
    Uses the model endpoint for generation.
    """
    if model is None:
        model = HF_GEN_MODEL
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code not in (200,201):
        raise RuntimeError(f"HF generate failed: {resp.status_code} {resp.text}")
    j = resp.json()
    # HF may return {'generated_text': '...'} or list of dicts [{'generated_text': '...'}] or plain text
    if isinstance(j, dict) and "generated_text" in j:
        return j["generated_text"]
    if isinstance(j, list) and len(j) > 0:
        # some endpoints return list of generations
        first = j[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        # sometimes it's {'generated_text': '...'} nested
        # or string
    if isinstance(j, str):
        return j
    # fallback: stringify
    return str(j)
