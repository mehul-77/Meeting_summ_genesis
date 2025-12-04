import React, { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [topK, setTopK] = useState(4)
  const [error, setError] = useState(null)

  async function handleFileChange(e) {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    await uploadAndProcess(f)
  }

  async function uploadAndProcess(f) {
    setError(null)
    setProcessing(true)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', f)

      const upRes = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
      if (!upRes.ok) throw new Error(`Upload failed: ${upRes.statusText}`)
      const upJson = await upRes.json()
      const file_id = upJson.file_id

      // process (RAG-enabled). include top_k
      const procRes = await fetch(`${API_BASE}/process/${file_id}?top_k=${encodeURIComponent(topK)}`, {
        method: 'POST'
      })
      if (!procRes.ok) {
        const txt = await procRes.text()
        throw new Error(`Processing failed: ${procRes.status} ${txt}`)
      }
      const out = await procRes.json()
      setResult(out)
    } catch (err) {
      console.error(err)
      setError(err.message)
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="container">
      <h1>Meeting Summarizer (RAG-enabled MVP)</h1>

      <div style={{ marginBottom: 12 }}>
        <label className="small">Select audio file (mp3/wav/m4a/ogg/webm):</label><br />
        <input type="file" accept="audio/*" onChange={handleFileChange} />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label className="small">Top K retrieved chunks (top_k): </label>
        <input type="number" min={1} max={12} value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: 64, marginLeft: 8 }} />
      </div>

      {processing && <p>Processing — this can take ~30s-2min depending on audio length and API latency...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 18 }}>
          <h2>Transcript</h2>
          <div className="pre">{result.transcript}</div>

          <h2 style={{ marginTop: 12 }}>Summary</h2>
          <div>{result.summary || 'No summary returned'}</div>

          <h2 style={{ marginTop: 12 }}>Action Items</h2>
          {Array.isArray(result.action_items) && result.action_items.length > 0 ? (
            <ul>
              {result.action_items.map((it, i) => {
                if (typeof it === 'string') return <li key={i}>{it}</li>
                return <li key={i}><strong>{it.task}</strong>{it.owner ? ` — ${it.owner}` : ''}{it.due_by ? ` (due: ${it.due_by})` : ''}</li>
              })}
            </ul>
          ) : <div>No structured action items returned.</div>}

          <h3 style={{ marginTop: 12 }}>Retrieved chunks (excerpts used for RAG)</h3>
          <div>
            {Array.isArray(result.retrieved_chunks) && result.retrieved_chunks.length > 0 ? (
              result.retrieved_chunks.map((c, idx) => (
                <div key={idx} style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: '#475569' }}>Chunk #{c.index}</div>
                  <div className="pre">{c.text_excerpt}</div>
                </div>
              ))
            ) : <div>No retrieved chunks.</div>}
          </div>

        </div>
      )}
    </div>
  )
}
