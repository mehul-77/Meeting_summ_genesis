import os
import uuid
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import requests
import chromadb
from chromadb.config import Settings

load_dotenv()

logger = logging.getLogger("uvicorn.error")

# ---------------- ENV ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    raise RuntimeError("GROQ_API_KEY missing or invalid")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- APP ----------------
app = FastAPI(title="Meeting Summarizer (Groq + RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CHROMA ----------------
chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PERSIST_DIR,
        anonymized_telemetry=False,
    )
)

# ---------------- HELPERS ----------------
def chunk_text(text, size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def groq_embeddings(texts):
    r = requests.post(
        "https://api.groq.com/openai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "text-embedding-3-small",
            "input": texts,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Embedding error: {r.text}")
    return [d["embedding"] for d in r.json()["data"]]

def groq_generate(prompt):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Generation error: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {"message": "Meeting Summarizer API is online", "docs": "/docs"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/__routes")
def list_routes():
    return [r.path for r in app.routes]

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    # Save with file_id prefix to find it later
    path = UPLOAD_DIR / f"{fid}-{file.filename}"
    async with aiofiles.open(path, "wb") as f:
        await f.write(await file.read())
    return {"file_id": fid}

@app.post("/process/{file_id}")
async def process(file_id: str, top_k: int = Query(4)):
    # 1. Locate the file
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(404, "File not found")

    audio_path = matches[0]

    # 2. Transcription
    try:
        with open(audio_path, "rb") as f:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": f},
                data={"model": "whisper-large-v3"},
                timeout=120,
            )
        if r.status_code != 200:
            raise HTTPException(500, f"Transcription failed: {r.text}")
        
        transcript = r.json()["text"]
    except Exception as e:
        raise HTTPException(500, str(e))

    # 3. RAG: Chunking and Vector Storage
    try:
        chunks = chunk_text(transcript)
        embeddings = groq_embeddings(chunks)

        collection = chroma_client.get_or_create_collection(name=f"coll_{file_id.replace('-', '_')}")
        ids = [f"{file_id}_{i}" for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
        )

        # Query using the transcript as the search context
        q_emb = groq_embeddings([transcript[:1000]])[0] # Use snippet for query embedding
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        context = "\n\n".join(results["documents"][0])
    except Exception as e:
        raise HTTPException(500, f"RAG Processing failed: {str(e)}")

    # 4. Generation
    prompt = f"""
    You are a meeting summarizer. 
    Return ONLY valid JSON with keys: "summary", "action_items", and "highlights".
    
    Context:
    {context}
    """

    try:
        raw_output = groq_generate(prompt)
        # Attempt to parse JSON from the LLM
        parsed = json.loads(raw_output)
    except Exception:
        # Fallback if LLM returns non-JSON text
        parsed = {"summary": raw_output, "action_items": [], "highlights": []}

    return {
        "file_id": file_id,
        "transcript": transcript,
        **parsed,
    }
