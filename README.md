# Meeting_summ_genesis

A SaaS platform that allows users to upload meeting audio recordings and automatically generates:

- Clean transcript  
- Concise meeting minutes  
- Actionable task lists  
- Grounded outputs using Retrieval-Augmented Generation (RAG)


---

##  Live Demo
🔗 **Live Product URL:** _<Add your deployed link here>_

🔗 **GitHub Repository:** _<Add repo URL once pushed>_

---

## Features
### **Core**
✔ Upload audio (mp3/wav/m4a/ogg/webm)  
✔ Whisper ASR transcription  
✔ RAG-powered summarisation  
✔ Action item extraction  
✔ JSON-structured output  
✔ Clean and simple React UI  

### **RAG Pipeline**
- Transcript is chunked with overlaps  
- OpenAI embeddings (`text-embedding-3-small`) generated for each chunk  
- FAISS vector index used for similarity search  
- Retrieved chunks from grounded LLM context  
- GPT-4o generates final summary + action items  

---

##  Tech Stack

### **Frontend**
- React + Vite  
- TailwindCSS (optional)  
- Fetch API for file uploads  
- Deployed on Vercel/Netlify  

### **Backend**
- FastAPI  
- Whisper ASR API  
- Groq API 
- RAG: FAISS + embeddings  
- Deployed on Render/Railway  

### **Other**
- Python  
- Node.js  


---


