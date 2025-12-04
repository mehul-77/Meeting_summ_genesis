# backend/rag_helpers.py
import math
from typing import List, Tuple
import numpy as np
import faiss
import openai

# Embedding model — adjust if needed
EMBED_MODEL = "text-embedding-3-small"

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 300) -> List[Tuple[str, int, int]]:
    """
    Split `text` into overlapping character chunks.
    Returns list of tuples: (chunk_text, start_char, end_char)
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for a list of texts using OpenAI embeddings endpoint.
    Returns list of embedding vectors (lists of floats).
    """
    embs = []
    # Use small batches if many texts
    for t in texts:
        resp = openai.Embedding.create(input=t, model=EMBED_MODEL)
        embs.append(resp['data'][0]['embedding'])
    return embs

def build_faiss_index(embeddings: List[List[float]]):
    """
    Build a simple Faiss IndexFlatL2 index from embeddings (float lists).
    Returns the faiss index and the numpy array (float32) of embeddings.
    """
    if len(embeddings) == 0:
        raise ValueError("No embeddings to build index.")
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    arr = np.array(embeddings).astype('float32')
    index.add(arr)
    return index, arr

def search_faiss(index, query_emb: List[float], top_k: int = 4):
    """
    Query the faiss index. Returns list of (index, distance).
    """
    q = np.array([query_emb]).astype('float32')
    D, I = index.search(q, top_k)
    indices = I[0].tolist()
    distances = D[0].tolist()
    return list(zip(indices, distances))

def assemble_context(chunks: List[Tuple[str,int,int]], indices: List[int], max_chars: int = 3500) -> str:
    """
    Concatenate selected chunk texts in order to create the context string for the LLM prompt.
    Limits total characters to max_chars.
    """
    parts = []
    total = 0
    for idx in indices:
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx][0]
        if total + len(c) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            parts.append(c[:remaining])
            total += remaining
            break
        parts.append(c)
        total += len(c)
    return "\n\n".join(parts)
