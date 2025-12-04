# backend/main.py
import os
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import requests
import aiofiles

# Load local .env for dev (ignored in prod)
load_dotenv()

# Environment variables (set these in Render / hosting env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")       # Groq transcription key (gsk_...)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # OpenAI classic/service key (sk-...)

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in env. Add it in Render / Env settings.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in env. Add it in Render / Env settings.")

# Create OpenAI v1 client
client = OpenAI(api_key=OPENAI_API_KEY)

# Upload directory
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Meeting Summarizer (Groq + OpenAI v1 RAG)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import rag helpers (expects get_embeddings(client, texts))
from rag_helpers import chunk_text, get_embeddings, build_faiss_index, search_faiss, assemble_context

ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".webm", ".ogg")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Save uploaded audio and return file_id."""
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use mp3/wav/m4a/ogg/webm.")
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
async def process_file(file_id: str, top_k: int = Query(4, description="Number of chunks to retrieve (RAG)")):
    """
    Process uploaded audio:
      1) Transcribe via Groq Whisper
      2) Chunk transcript
      3) Create embeddings via OpenAI v1 client
      4) Build FAISS index, retrieve top_k
      5) Run chat completion (OpenAI v1) with retrieved context
    """
    # locate file
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    audio_path = matches[0]

    # -------------------------
    # 1) Transcribe using Groq
    # -------------------------
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

    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Groq transcription failed: {resp.status_code} {resp.text}")

    j = resp.json()
    transcript_text = j.get("text") or ""
    if not transcript_text.strip():
        raise HTTPException(status_code=500, detail="Transcription returned empty text.")

    # -------------------------
    # 2) Chunk the transcript
    # -------------------------
    chunks = chunk_text(transcript_text, chunk_size=1200, overlap=300)
    chunk_texts = [c[0] for c in chunks]

    # -------------------------
    # 3) Get embeddings (OpenAI v1 client)
    # -------------------------
    try:
        embeddings = get_embeddings(client, chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embeddings error: {e}")

    # -------------------------
    # 4) Build FAISS index
    # -------------------------
    try:
        index, _arr = build_faiss_index(embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS build error: {e}")

    # -------------------------
    # 5) Query embedding & retrieval
    # -------------------------
    try:
        q_resp = client.embeddings.create(model="text-embedding-3-small", input=transcript_text)
        try:
            query_embed = q_resp.data[0].embedding
        except Exception:
            query_embed = q_resp["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding error: {e}")

    try:
        hits = search_faiss(index, query_embed, top_k=top_k)
        retrieved_indices = [h[0] for h in hits]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search error: {e}")

    context_str = assemble_context(chunks, retrieved_indices, max_chars=3500)

    # -------------------------
    # 6) LLM summarization (OpenAI v1 chat)
    # -------------------------
    system_prompt = (
        "You are an AI meeting summarizer. Use ONLY the provided context to produce a JSON object with fields:"
        " summary (3-5 sentences), action_items (array of objects with keys: task (string), owner (optional string), due_by (optional string)),"
        " and highlights (array of 3 short strings). DO NOT hallucinate facts or invent names/dates. If owner/due_by unknown, leave blank or null."
    )
    user_prompt = f"Context (relevant transcript chunks):\n\n{context_str}\n\nProduce strictly valid JSON."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",  # change to "gpt-3.5-turbo" for cheaper inference if desired
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=700,
            temperature=0.0,
        )
        try:
            raw = resp.choices[0].message.content
        except Exception:
            raw = resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {e}")

    # parse JSON; fallback to raw
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"summary": raw, "action_items": [], "highlights": []}

    response = {
        "file_id": file_id,
        "transcript": transcript_text,
        "summary": parsed.get("summary"),
        "action_items": parsed.get("action_items"),
        "highlights": parsed.get("highlights"),
        "retrieved_chunks": [
            {"index": i, "text_excerpt": chunks[i][0][:300]} for i in retrieved_indices if 0 <= i < len(chunks)
        ]
    }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
