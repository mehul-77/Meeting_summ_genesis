// frontend/src/components/MeetingView.jsx
import React from "react";

export default function MeetingView({ meeting, audioUrl }) {
  const { transcript, summary, action_items, file_name } = meeting || {};

  return (
    <div style={{ marginTop: 16 }}>
      <div className="card">
        <h3>Audio</h3>
        {audioUrl ? <audio controls src={audioUrl} /> : <div>No audio URL</div>}
        <div style={{ fontSize: 12, color: "#555", marginTop: 6 }}>{file_name}</div>
      </div>

      <div className="card">
        <h3>Transcript</h3>
        <pre style={{ whiteSpace: "pre-wrap" }}>{transcript || "—"}</pre>
      </div>

      <div className="card">
        <h3>Summary</h3>
        <div>{summary || "—"}</div>
      </div>

      <div className="card">
        <h3>Action items</h3>
        {action_items && action_items.length ? (
          <ul>{action_items.map((a) => <li key={a.id || a.text}>{a.text || JSON.stringify(a)}</li>)}</ul>
        ) : <div>None</div>}
      </div>
    </div>
  );
}
