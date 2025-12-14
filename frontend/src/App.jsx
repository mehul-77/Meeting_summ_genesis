import React, { useState } from 'react'
import FileUploader from './components/FileUploader';
import MeetingView from './components/MeetingView';

const API_BASE = import.meta.env.VITE_API_BASE;

export default function App() {
  const [meeting, setMeeting] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  function handleUploaded(info) {
    const id = info.file_id || info.fileId;
    setFileId(id);
    setAudioUrl(`${API_BASE}/files/${id}`);
  }

  async function handleProcess() {
    if (!fileId) {
      alert("Please upload first!");
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/process/${fileId}?top_k=4`, {
        method: "POST",
        headers: { "Accept": "application/json" }
      });

      const text = await res.text();
      let json;

      try {
        json = JSON.parse(text);
      } catch {
        alert("Invalid backend response:\n" + text);
        return;
      }

      if (res.ok) {
        setMeeting(json);
      } else {
        alert(JSON.stringify(json));
      }

    } catch (err) {
      alert("Process failed: " + err.message);
    }
  }

  return (
    <div className="container">
      <h1>Meeting Summarizer</h1>

      <FileUploader 
        onUploaded={handleUploaded}
        apiBase={API_BASE}
      />

      <div style={{ marginTop: 12 }}>
        <button 
          onClick={handleProcess} 
          className="btn" 
          disabled={!fileId}
        >
          Process uploaded file
        </button>
      </div>

      {meeting && <MeetingView meeting={meeting} audioUrl={audioUrl} />}
    </div>
  );
}
