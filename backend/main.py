# backend/main.py
import os
import uuid
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import aiofiles

# load local .env if present (only for local dev)
load_dotenv()

# env vars (set in Render)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in env. Set it in Render env variables.")
if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY in env. Set it in Render env variables.")

# upload dir
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Meeting Summarizer (Groq + Cohere RAG + Chroma)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# import rag helpers
from rag_helpers import chunk_text, cohere_embed_texts, build_chroma_collection, upsert_embeddings_to_chroma, query_chroma

ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".webm", ".ogg")

# Cohere generation model and params
COHERE_GEN_MODEL = os.getenv("COHERE_GEN_MODEL", "command-xlarge-nightly")  # you can change to another model available in your account
COHERE_GEN_MAX_TOKENS = int(os.getenv("COHERE_GEN_MAX_TOKENS", "512"))

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
async def process_file(file_id: str, top_k: int = Query(4, description="Number of chunks to retrieve for RAG")):
    # locate file
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    audio_path = matches[0]

    # 1) Transcribe via Groq Whisper
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

    # 2) Chunk transcript
    chunks = chunk_text(transcript_text, chunk_size=1000, overlap=250)
    chunk_texts = [c[0] for c in chunks]

    # 3) Create embeddings via Cohere
    try:
        embeddings = cohere_embed_texts(COHERE_API_KEY, chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere embeddings failed: {e}")

    # 4) Create or reuse chroma collection and upsert embeddings
    try:
        chroma_client, collection = build_chroma_collection(collection_name=file_id, persist=True, client=None)
        # Prepare ids
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunk_texts))]
        metadatas = [{"start": chunks[i][1], "end": chunks[i][2], "source": file_id} for i in range(len(chunks))]
        upsert_embeddings_to_chroma(collection, ids=ids, texts=chunk_texts, embeddings=embeddings, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma upsert error: {e}")

    # 5) Query chroma with embedding of the whole transcript to get relevant chunks
    try:
        query_embed_resp = cohere_embed_texts(COHERE_API_KEY, [transcript_text])
        query_embed = query_embed_resp[0]
        results = query_chroma(collection, query_embed, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma query / embedding error: {e}")

    # assemble retrieved context
    retrieved_texts = [r["document"] for r in results]
    context_str = "\n\n".join(retrieved_texts)

    # 6) Use Cohere generate API to produce summary + action items based on retrieved context
    system_instructions = (
        "You are an assistant that, given meeting transcript context, returns a JSON object with fields:\n"
        "summary: concise paragraph (3-5 sentences), action_items: list of {task, owner (optional), due_by (optional)}, highlights: 3 bullets.\n"
        "Only use information present in the context. Return strictly valid JSON."
    )
    prompt = f"{system_instructions}\n\nContext:\n{context_str}\n\nReturn JSON only."

    try:
        gen_url = "https://api.cohere.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        gen_payload = {
            "model": COHERE_GEN_MODEL,
            "prompt": prompt,
            "max_tokens": COHERE_GEN_MAX_TOKENS,
            "temperature": 0.0,
            "stop_sequences": ["\n\n"]
        }
        gen_resp = requests.post(gen_url, json=gen_payload, headers=headers, timeout=120)
        if gen_resp.status_code not in (200,201):
            raise HTTPException(status_code=500, detail=f"Cohere generate failed: {gen_resp.status_code} {gen_resp.text}")
        gen_json = gen_resp.json()
        # response structure: gen_json['generations'][0]['text'] or gen_json['text'] depending on API version
        generated_text = None
        if "generations" in gen_json and isinstance(gen_json["generations"], list) and len(gen_json["generations"])>0:
            generated_text = gen_json["generations"][0].get("text") or gen_json["generations"][0].get("generation", {}).get("text")
        elif "output" in gen_json:
            # some API variants
            generated_text = gen_json["output"][0] if isinstance(gen_json["output"], list) else gen_json["output"]
        else:
            # fallback to entire text
            generated_text = gen_json.get("text") or str(gen_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere generate error: {e}")

    # parse JSON produced by the model
    try:
        parsed = json.loads(generated_text)
    except Exception:
        parsed = {"summary": generated_text, "action_items": [], "highlights": []}

    # build response
    response = {
        "file_id": file_id,
        "transcript": transcript_text,
        "summary": parsed.get("summary"),
        "action_items": parsed.get("action_items"),
        "highlights": parsed.get("highlights"),
        "retrieved_chunks": [{"id": r["id"], "distance": r["distance"], "text": r["document"][:400]} for r in results]
    }

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
