# backend/main.py
import os
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import openai
import aiofiles

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables.")
openai.api_key = OPENAI_API_KEY

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Meeting Summarizer (RAG-enabled)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import rag helpers
from rag_helpers import chunk_text, get_embeddings, build_faiss_index, search_faiss, assemble_context

ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".webm", ".ogg")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use mp3/wav/m4a/ogg/webm.")
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}-{file.filename}"
    try:
        async with aiofiles.open(dest, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    return {"file_id": file_id, "filename": file.filename, "path": str(dest)}

@app.post("/process/{file_id}")
async def process_file(file_id: str, top_k: int = Query(4, description="Number of chunks to retrieve")):
    # Locate file
    matches = list(UPLOAD_DIR.glob(f"{file_id}-*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    audio_path = matches[0]

    # 1) Transcribe with Whisper (synchronous call)
    try:
        with open(audio_path, "rb") as af:
            transcription = openai.Audio.transcribe("whisper-1", af)
            transcript_text = transcription.get("text", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

    # 2) Chunk transcript
    chunks = chunk_text(transcript_text, chunk_size=1200, overlap=300)
    chunk_texts = [c[0] for c in chunks]

    # 3) Create embeddings for chunks
    try:
        embeddings = get_embeddings(chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embeddings error: {e}")

    # 4) Build FAISS index
    try:
        index, arr = build_faiss_index(embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS build error: {e}")

    # 5) Create query embedding; use a short focused query for better retrieval or use the transcript
    try:
        # Use the whole transcript as the query to find globally relevant chunks
        query_embed = get_embeddings([transcript_text])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding error: {e}")

    # 6) Retrieve top_k chunk indices
    try:
        hits = search_faiss(index, query_embed, top_k=top_k)
        retrieved_indices = [h[0] for h in hits]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search error: {e}")

    # 7) Assemble retrieved context
    context_str = assemble_context(chunks, retrieved_indices, max_chars=3500)

    # 8) Prompt LLM with retrieved context
    system_prompt = (
        "You are an AI meeting summarizer. Use ONLY the provided context to produce a JSON object with fields:"
        " summary (3-5 sentences), action_items (array of objects with keys: task (string), owner (optional string), due_by (optional string)),"
        " and highlights (array of 3 short strings). DO NOT hallucinate facts or invent names/dates. If owner/due_by unknown, leave them blank or null."
    )
    user_prompt = f"Context (relevant transcript chunks):\n\n{context_str}\n\nProduce strictly valid JSON."

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",  # change to gpt-3.5-turbo for cheaper runs if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=700,
            temperature=0.0,
        )
        raw = resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {e}")

    # Try parse JSON; fallback to raw
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
