import os
import uuid
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

# Load embedding model ONCE (important)
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Local embeddings – FREE, no API
    """
    vectors = _embedding_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return vectors.tolist()


def build_chroma_collection(collection_name="meetings"):
    client = chromadb.Client(
        Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        )
    )

    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)

    return client, collection


def add_chunks(collection, texts: List[str], metadatas: List[dict]):
    ids = [str(uuid.uuid4()) for _ in texts]
    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


def query_chunks(collection, query: str, top_k: int = 4):
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0] if results["documents"] else []
