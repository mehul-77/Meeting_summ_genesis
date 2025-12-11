# backend/main.py
import os
import uuid
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import aiofiles

# Load local .env for development only
load_dotenv()

logger = logging.getLogger("uvicorn.error")

# Env vars (set these in Render)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face token
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"))

# Basic sanity checks (fail fast on startup)
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set. Transcription will fail until you add it to environment.")
if GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_"):
    logger.warning("GROQ_API_KEY does not start with 'gsk_'. Verify you created a User API Key (not a service-account key).")

if not HF_API_KEY:
    logger.warning("HF_API_KEY is not set. Embeddings/generation will fail until you add it to environment.")

# Ensure upload dir exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Meeting Summarizer — Groq + HF RAG + Chroma (backend)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import rag helpers (these must exist in backend/rag_helpers.py)
from rag_helpers import (
    chunk_text,
    hf_embeddings,
    build_chroma_collection,
    upsert_embeddings_to_chroma,
    query_chroma,
    hf_generate,
)

# Probe the vector store once at startup to log which backend is used (chromadb or fallback)
try:
    _client_probe, _collection_probe = build_chroma_collection(collection_name="startup-probe", persist=False)
    logger.info("RAG helper: build_chroma_collection succeeded (chromadb or fallback available).")
except Exception as e:
    logger.exception("RAG helper startup probe failed: %s", e)

ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".webm", ".ogg")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Save uploaded audio and return file_id to process later.
    """
    filename = file.filename or "upload"
    if not filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use mp3/wav/m4a/ogg/webm.")
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}-{filename}"
    try:
        async with aiofiles.open(dest, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        logger.exception("Failed to save uploaded file: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    return {"file_id": file_id, "filename": filename}


@app.post("/process/{file_id}")
async def process_file(file_id: str, top_k: int = Query(4, description="Number of chunks to retrieve for RAG")):
    """
    Process uploaded audio:
      1) Transcribe via Groq Whisper
      2) Chunk transcript
      3) Create embeddings via Hugging Face
      4) Upsert into Chroma (or in-memory fallback)
      5) Retrieve top_k chunks
      6) Generate JSON summary + action items using HF generation model
    """
    # Locate audio file
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    audio_path = matches[0]

    # 1) Transcribe using Groq (fast fail if key not present)
    if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
        raise HTTPException(status_code=500, detail="Groq API key missing or invalid. Set GROQ_API_KEY (user API key starting with 'gsk_') in environment.")

    try:
        with open(audio_path, "rb") as f:
            groq_resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": f},
                data={"model": "whisper-large-v3"},
                timeout=120
            )
    except requests.RequestException as e:
        logger.exception("Groq transcription request failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Groq transcription request failed: {e}")

    if groq_resp.status_code not in (200, 201):
        # forward Groq error message for debugging
        logger.error("Groq transcription failed: %s %s", groq_resp.status_code, groq_resp.text)
        raise HTTPException(status_code=500, detail=f"Groq transcription failed: {groq_resp.status_code} {groq_resp.text}")

    groq_json = groq_resp.json()
    transcript_text = groq_json.get("text") or ""
    if not transcript_text.strip():
        logger.error("Transcription returned empty text for file_id %s", file_id)
        raise HTTPException(status_code=500, detail="Transcription returned empty text.")

    # 2) Chunk transcript
    chunks = chunk_text(transcript_text, chunk_size=800, overlap=200)
    chunk_texts = [c[0] for c in chunks]

    # 3) Get embeddings for chunks using HF
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HF API key missing. Set HF_API_KEY in environment for embeddings/generation.")
    try:
        embeddings = hf_embeddings(HF_API_KEY, chunk_texts)
    except Exception as e:
        logger.exception("HF embeddings failed: %s", e)
        raise HTTPException(status_code=500, detail=f"HF embeddings failed: {e}")

    # 4) Build or get chroma collection and upsert embeddings
    try:
        chroma_client, collection = build_chroma_collection(collection_name=file_id, persist=True)
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunk_texts))]
        metadatas = [{"start": chunks[i][1], "end": chunks[i][2], "source": file_id} for i in range(len(chunks))]
        upsert_embeddings_to_chroma(collection, ids=ids, texts=chunk_texts, embeddings=embeddings, metadatas=metadatas)
    except Exception as e:
        logger.exception("Chroma upsert error: %s", e)
        raise HTTPException(status_code=500, detail=f"Chroma upsert error: {e}")

    # 5) Query: embed full transcript and retrieve
    try:
        query_embs = hf_embeddings(HF_API_KEY, [transcript_text])
        query_emb = query_embs[0]
        results = query_chroma(collection, query_emb, top_k=top_k)
    except Exception as e:
        logger.exception("Chroma query / embedding error: %s", e)
        raise HTTPException(status_code=500, detail=f"Chroma query / embedding error: {e}")

    retrieved_texts = [r["document"] for r in results]
    context_str = "\n\n".join(retrieved_texts) if retrieved_texts else transcript_text[:3000]

    # 6) Generate using HF generation endpoint
    prompt = (
        "You are a meeting summarizer. Use ONLY the provided context to produce a JSON object with keys:\n"
        "summary: concise paragraph (3-5 sentences),\n"
        "action_items: list of {task: string, owner: optional string, due_by: optional string},\n"
        "highlights: list of 3 short strings.\n"
        "Do not hallucinate. Only use facts in the context.\n\n"
        f"Context:\n{context_str}\n\nReturn strictly valid JSON."
    )

    try:
        generated_text = hf_generate(HF_API_KEY, prompt, max_tokens=400, temperature=0.0)
    except Exception as e:
        logger.exception("HF generate error: %s", e)
        raise HTTPException(status_code=500, detail=f"HF generate error: {e}")

    # parse JSON (fallback to raw text)
    try:
        parsed = json.loads(generated_text)
    except Exception:
        logger.warning("Failed to parse JSON from model output; returning raw generated text in 'summary'.")
        parsed = {"summary": generated_text, "action_items": [], "highlights": []}

    response = {
        "file_id": file_id,
        "transcript": transcript_text,
        "summary": parsed.get("summary"),
        "action_items": parsed.get("action_items"),
        "highlights": parsed.get("highlights"),
        "retrieved_chunks": [{"id": r["id"], "distance": r["distance"], "text": r["document"][:400]} for r in results]
    }

    return response


@app.get("/healthz")
def healthz():
    return {"status": "ok", "upload_dir_exists": UPLOAD_DIR.exists(), "chroma_persist_dir": CHROMA_PERSIST_DIR}


if __name__ == "__main__":
    import uvicorn
    # run with: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
