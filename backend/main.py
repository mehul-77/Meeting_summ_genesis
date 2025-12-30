import os
import uuid
import json
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import requests
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

load_dotenv()

# ---------------- CONFIG ----------------
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is missing!")
    
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- APP ----------------
app = FastAPI(title="Meeting Summarizer (Groq + Local Embeddings)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CHROMA SETUP ----------------
# Use a lighter, ephemeral client for Render Free Tier to avoid file lock issues
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# Use built-in SentenceTransformer (all-MiniLM-L6-v2) - runs locally, no API key needed
# Note: On Render Free Tier (512MB RAM), this is tight but usually works.
default_ef = embedding_functions.DefaultEmbeddingFunction()

# ---------------- HELPERS ----------------
def chunk_text(text, size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def groq_generate(prompt):
    """Call Groq Llama 3 for the final summary"""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq Generation Error: {str(e)}")
        raise e

# ---------------- ROUTES ----------------
from fastapi.responses import FileResponse

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(404, "File not found")
    return FileResponse(matches[0])

@app.get("/")

def home():
    return {"status": "online", "message": "Go to /docs to test the API"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Uploads the audio file to the server."""
    try:
        fid = str(uuid.uuid4())
        path = UPLOAD_DIR / f"{fid}-{file.filename}"
        async with aiofiles.open(path, "wb") as f:
            await f.write(await file.read())
        return {"file_id": fid, "filename": file.filename}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/process/{file_id}")
async def process(file_id: str, top_k: int = Query(4)):
    """Transcribes audio, chunks text, and generates a summary using RAG."""
    
    # 1. FIND FILE
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(404, "File not found")
    audio_path = matches[0]

    # 2. TRANSCRIBE (Groq Whisper)
    transcript = ""
    try:
        logger.info(f"Starting transcription for {file_id}...")
        with open(audio_path, "rb") as f:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": f},
                data={"model": "whisper-large-v3"},
                timeout=120,
            )
        r.raise_for_status()
        transcript = r.json()["text"]
        logger.info("Transcription complete.")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    # 3. RAG (Local Embeddings + Chroma)
    try:
        logger.info("Starting RAG processing...")
        chunks = chunk_text(transcript)
        
        # Unique collection name (sanitized)
        coll_name = f"session_{file_id.replace('-', '')}"
        
        # Reset collection if it exists
        try:
            chroma_client.delete_collection(name=coll_name)
        except:
            pass
            
        collection = chroma_client.create_collection(
            name=coll_name,
            embedding_function=default_ef
        )

        ids = [f"id_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)

        # Query relevant chunks (using the first 500 chars as a proxy query)
        results = collection.query(
            query_texts=[transcript[:500]], 
            n_results=min(top_k, len(chunks))
        )
        context = "\n\n".join(results["documents"][0])
        logger.info("RAG context retrieved.")
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        # Fallback: if RAG fails (e.g. memory issues), use raw transcript
        context = transcript[:15000] # Truncate to be safe

    # 4. SUMMARIZE (Groq Llama 3)
    prompt = f"""
    You are a helpful assistant. Summarize the following meeting transcript.
    Format the output as valid JSON with keys: "summary", "action_items", "highlights".
    
    Transcript Context:
    {context}
    """
    
    try:
        raw_response = groq_generate(prompt)
        # Attempt to parse JSON; fallback to raw text if model chats too much
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to find JSON inside code blocks if the model wrapped it
            if "```json" in raw_response:
                import re
                match = re.search(r"```json(.*?)```", raw_response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                else:
                    parsed = {"summary": raw_response, "action_items": [], "highlights": []}
            else:
                parsed = {"summary": raw_response, "action_items": [], "highlights": []}
                
        return {
            "file_id": file_id,
            "transcript_snippet": transcript[:200] + "...",
            **parsed
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")
