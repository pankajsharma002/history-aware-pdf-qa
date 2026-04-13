# History-Aware PDF Question Answering

A history-aware, conversational Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and chat with their content.
Built using Streamlit, LangChain, FAISS, HuggingFace embeddings, Groq LLM, and a Cross-Encoder reranker, with full chat history support.

---

## 🚀 Features

- Upload PDF files and split them into semantic chunks
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- History-aware conversational RAG
- Reranking using Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Fast inference using Groq LLM (`llama-3.1-8b-instant`)
- Session-based chat history
- Simple and interactive Streamlit UI
- Fully Dockerized for easy deployment

---

## 🧱 Project Structure


rag_pdf_chat/
├── app/
│   ├── main.py              # Streamlit application
│   ├── requirements.txt     # Python dependencies
├── data/                    # (Optional) Temporary PDF storage
├── .env                     # API keys (not committed)
├── Dockerfile               # Docker configuration
├── .dockerignore            # Files ignored by Docker
└── README.md                # Project documentation
---
## Architecure Overview


User Query
   ↓
Query Rewriting (History-Aware)
   ↓
FAISS Retriever (Top K = 10)
   ↓
Cross-Encoder Reranker
   ↓
Top 3 Relevant Chunks
   ↓
LLM (Groq - LLaMA 3.1)
   ↓
Final Answer

---

> This project uses **in-memory FAISS vector storage**, so no `vector_store` directory is required.

---

## 🔧 Tech Stack

- Frontend: Streamlit
- LLM: Groq (LLaMA 3.1)
- Embeddings: HuggingFace Sentence Transformers
- Vector Database: FAISS (in-memory)
- Framework: LangChain
- Containerization: Docker

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/pankajsharma002/history-aware-pdf-qa.git
cd history-aware-pdf-qa
