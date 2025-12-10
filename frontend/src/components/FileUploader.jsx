// frontend/src/components/FileUploader.jsx
import React, { useRef, useState } from "react";

export default function FileUploader({ onUploaded, apiBase }) {
  const ref = useRef();
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);

  async function uploadFile(file) {
    setStatus("uploading");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${apiBase}/upload`, { method: "POST", body: fd });
      if (!res.ok) {
        setStatus("error");
        const txt = await res.text();
        return alert("Upload failed: " + txt);
      }
      const j = await res.json();
      setStatus("done");
      setProgress(100);
      onUploaded && onUploaded(j);
    } catch (err) {
      setStatus("error");
      alert("Upload error: " + err.message);
    }
  }

  function onFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    uploadFile(f);
  }

  function onDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) uploadFile(f);
  }

  return (
    <div onDrop={onDrop} onDragOver={(e) => e.preventDefault()} className="uploader">
      <input ref={ref} type="file" accept="audio/*" onChange={onFile} style={{ display: "none" }} />
      <button onClick={() => ref.current.click()} className="btn">
        Pick audio file
      </button>
      <div style={{ marginTop: 8 }}>
        <div>Status: {status} {progress > 0 && <>— {progress}%</>}</div>
      </div>
      <div className="hint">or drag & drop an audio file here</div>
    </div>
  );
}
