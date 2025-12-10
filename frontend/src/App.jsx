import React, { useState } from 'react'
import FileUploader from './components/FileUploader';
import MeetingView from './components/MeetingView';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [meeting, setMeeting] = useState(null)
  const [fileId, setFileId] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)

  function handleUploaded(info) {
    const id = info.file_id || info.fileId || info.fileId;
    setFileId(id)
    setAudioUrl(`${API_BASE}/files/${id}`)
  }

  async function handleProcess(){
    if(!fileId) return alert("Upload a file first");
    const res = await fetch(`${API_BASE}/process/${fileId}`);

    if (!res.ok) {
      const txt = await res.text();
      return alert(`Error processing file:` + txt);
    }
    const j = await res.json();
    setMeeting(j);
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
