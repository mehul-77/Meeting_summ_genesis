# backend/main.py
import os
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import aiofiles

load_dotenv()

# env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face token, free tier available
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")
if not HF_API_KEY:
    raise RuntimeError("Missing HF_API_KEY in environment. Get a free token from https://huggingface.co/settings/tokens")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Meeting Summarizer — Groq + HF RAG + Chroma")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# helpers
from rag_helpers import chunk_text, hf_embeddings, build_chroma, upsert_chroma, query_chroma, hf_generate

ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".webm", ".ogg")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}-{file.filename}"
    try:
        async with aiofiles.open(dest, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    return {"file_id": file_id, "filename": file.filename}

@app.post("/process/{file_id}")
async def process_file(file_id: str, top_k: int = Query(4, description="top_k for retrieval")):
    # find file
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    audio_path = matches[0]

    # 1) Groq transcription
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": f},
                data={"model": "whisper-large-v3"},
                timeout=120
            )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Groq transcription request failed: {e}")

    if resp.status_code not in (200,201):
        raise HTTPException(status_code=500, detail=f"Groq transcription failed: {resp.status_code} {resp.text}")

    j = resp.json()
    transcript_text = j.get("text") or ""
    if not transcript_text.strip():
        raise HTTPException(status_code=500, detail="Transcription returned empty text.")

    # 2) chunk
    chunks = chunk_text(transcript_text, chunk_size=800, overlap=200)
    texts = [c[0] for c in chunks]

    # 3) get embeddings from HF
    try:
        embeddings = hf_embeddings(HF_API_KEY, texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF embeddings error: {e}")

    # 4) upsert into chroma (collection per file_id)
    try:
        client, collection = build_chroma(collection_name=file_id, persist=True)
        ids = [f"{file_id}_chunk_{i}" for i in range(len(texts))]
        metadatas = [{"start": chunks[i][1], "end": chunks[i][2]} for i in range(len(chunks))]
        upsert_chroma(collection, ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma upsert error: {e}")

    # 5) query: embed full transcript and retrieve
    try:
        q_emb = hf_embeddings(HF_API_KEY, [transcript_text])[0]
        results = query_chroma(collection, q_emb, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")

    retrieved_texts = [r["document"] for r in results]
    context = "\n\n".join(retrieved_texts)

    # 6) Generate summary + actions using HF generation model
    prompt = (
        "You are a meeting summarizer. Use ONLY the provided context to output a JSON object with keys:\n"
        "summary: 3-5 sentence concise summary,\n"
        "action_items: array of objects {task: string, owner: optional, due_by: optional},\n"
        "highlights: array of 3 short bullet strings.\n"
        "Do NOT hallucinate. Only use facts in the context.\n\n"
        f"Context:\n{context}\n\nOutput strictly valid JSON."
    )

    try:
        generated = hf_generate(HF_API_KEY, prompt, max_tokens=400, temperature=0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF generation error: {e}")

    # parse JSON
    try:
        parsed = json.loads(generated)
    except Exception:
        # fallback: return raw text in summary
        parsed = {"summary": generated, "action_items": [], "highlights": []}

    response = {
        "file_id": file_id,
        "transcript": transcript_text,
        "summary": parsed.get("summary"),
        "action_items": parsed.get("action_items"),
        "highlights": parsed.get("highlights"),
        "retrieved_chunks": [{"id": r["id"], "distance": r["distance"], "text": r["document"][:400]} for r in results]
    }
    return response
