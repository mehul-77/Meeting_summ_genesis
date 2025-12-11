import React, { useState } from 'react'
import FileUploader from './components/FileUploader';
import MeetingView from './components/MeetingView';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const BACKEND = import.meta.env.VITE_API_BASE;
export default function App() {
  const [meeting, setMeeting] = useState(null)
  const [fileId, setFileId] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)

  function handleUploaded(info) {
    const id = info.file_id || info.fileId || info.fileId;
    setFileId(id)
    setAudioUrl(`${API_BASE}/files/${id}`)
  }

 async function handleProcess() {
  if (!uploadedFileId) {
    alert("Please upload first!");
    return;
  }

  try {
    const res = await fetch(`${BACKEND}/process/${uploadedFileId}?top_k=4`, {
      method: "POST",
      headers: { "Accept": "application/json" }
    });

    const txt = await res.text();
    const json = JSON.parse(txt);
    console.log(json);
  } catch (err) {
    alert("Error: " + err.message);
  }
}
  return (
    <div className="container">
      <h1>Meeting Summarizer</h1>
      <FileUploader onUploaded={handleUploaded} 
      apiBase = {API_BASE}/>
      <div style={{marginTop:12}}>
        <button onClick={handleProcess} className='btn' disabled={!fileId}>
          Process uploaded file
        </button>
      </div>

      {meeting && <MeetingView meeting={meeting} audioUrl={audioUrl} />}
      
    </div>
  );
}
