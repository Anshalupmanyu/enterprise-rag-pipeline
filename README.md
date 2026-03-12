# 🏢 Enterprise RAG Pipeline

> Production-ready Retrieval-Augmented Generation system using open-source LLMs — fully containerized at **zero cost**.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange.svg)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This project implements a **production-ready RAG (Retrieval-Augmented Generation)** system designed to mirror enterprise Generative AI service architecture patterns (e.g., OCI GenAI). It combines:

- **Mistral 7B** via Ollama — high-quality open-source LLM, zero API cost
- **FAISS** — Facebook AI Similarity Search for sub-millisecond vector retrieval
- **LangChain** — orchestration framework connecting retrieval to generation
- **HuggingFace** — `all-MiniLM-L6-v2` sentence embeddings
- **FastAPI** — async REST API with automatic OpenAPI documentation
- **JWT Authentication** — secure, stateless token-based auth
- **Docker** — fully containerized, one-command deployment

**Target: sub-800ms end-to-end query response latency.**

---

## 🏗️ Architecture

```
┌────────────────┐     ┌───────────────────────┐     ┌──────────────────┐
│                │     │    FastAPI REST API    │     │   Ollama Server  │
│  Client / cURL │────▶│  ┌─────────────────┐  │────▶│   (Mistral 7B)   │
│                │◀────│  │  JWT Auth Guard  │  │◀────│                  │
└────────────────┘     │  └─────────────────┘  │     └──────────────────┘
                       │  ┌─────────────────┐  │
                       │  │  RAG Chain       │  │
                       │  │  (LangChain)     │  │
                       │  └────────┬────────┘  │
                       │           │           │
                       │  ┌────────▼────────┐  │
                       │  │  FAISS Vector   │  │
                       │  │  Store + HF     │  │
                       │  │  Embeddings     │  │
                       │  └─────────────────┘  │
                       └───────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- ~4 GB disk space (for Mistral model + embeddings)

### 1. Clone & Configure

```bash
git clone <your-repo-url>
cd enterprise-rag-pipeline
cp .env.example .env
# Edit .env to set a strong JWT_SECRET_KEY
```

### 2. Launch with Docker Compose

```bash
docker-compose up -d --build
```

This starts:
| Service | Description | Port |
|---------|-------------|------|
| `ollama` | LLM server (auto-pulls Mistral) | 11434 |
| `rag-api` | FastAPI application | 8000 |

> ⏳ First launch takes 5-10 min to pull the Mistral model 

### 3. Use the API

```bash
# Register a user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "securepass123"}'

# Get JWT token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=securepass123" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Ingest documents
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "Authorization: Bearer $TOKEN"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key components of a RAG architecture?"}'
```

### 4. Interactive Docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/auth/register` | ❌ | Register a new user |
| `POST` | `/api/v1/auth/token` | ❌ | Login → JWT token |

### RAG Query

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/query` | ✅ JWT | Ask a question (RAG) |
| `GET` | `/api/v1/query/health` | ❌ | Pipeline health check |

### Documents

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/documents/ingest` | ✅ JWT | Trigger ingestion/indexing |
| `GET` | `/api/v1/documents/stats` | ✅ JWT | Vector store statistics |

### Example Query Response

```json
{
  "answer": "The key components of a RAG architecture include: 1) Document Ingestion Pipeline for parsing and chunking documents, 2) Vector Store (FAISS) for storing embeddings and performing similarity search, 3) Retrieval Strategy with configurable top-k retrieval, and 4) Generation Layer using an LLM to produce grounded responses with citations.",
  "sources": [
    {
      "filename": "enterprise_ai_overview.txt",
      "chunk_index": 5,
      "excerpt": "RAG has emerged as the dominant enterprise pattern for leveraging Large Language Models..."
    }
  ],
  "latency_ms": 542.3,
  "model": "mistral"
}
```

---

## 📁 Project Structure

```
enterprise-rag-pipeline/
├── app/
│   ├── main.py                 # FastAPI app factory + lifespan
│   ├── core/
│   │   ├── config.py           # Pydantic Settings (env vars)
│   │   └── security.py         # JWT + bcrypt authentication
│   ├── services/
│   │   ├── ingestion.py        # Document loading & chunking
│   │   ├── vector_store.py     # FAISS index management
│   │   └── rag_chain.py        # LangChain RetrievalQA + Ollama
│   └── api/
│       ├── auth.py             # Auth routes (register/login)
│       ├── query.py            # RAG query + health routes
│       └── documents.py        # Ingestion + stats routes
├── data/
│   └── documents/              # Source documents for ingestion
│       ├── enterprise_ai_overview.txt
│       └── oci_generative_ai.txt
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Ollama + RAG API orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md
```

---

## ⚙️ Configuration

All settings are configurable via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `LLM_MODEL` | `mistral` | Model name for generation |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `SIMILARITY_TOP_K` | `4` | Retrieved chunks per query |
| `JWT_SECRET_KEY` | *(change me)* | JWT signing secret |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token lifetime |

---

## 🔧 Local Development (without Docker)

```bash
# 1. Install Ollama (https://ollama.ai)
ollama pull mistral

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Set OLLAMA_BASE_URL=http://localhost:11434

# 5. Run
uvicorn app.main:app --reload --port 8000
```

---

## 🧪 Performance

| Metric | Target | Design Choice |
|--------|--------|---------------|
| End-to-end latency | < 800ms | Optimized chunking + FAISS ANN search |
| Vector search | < 20ms | FAISS with L2 distance, in-memory index |
| Embedding generation | < 50ms | MiniLM-L6-v2 (22M params, CPU-friendly) |
| Authentication | < 5ms | Stateless JWT verification |
| Startup time | < 30s | Pre-downloaded model in Docker image |

---

## 🛡️ Security Features

- **JWT Authentication** — Stateless, signed tokens with configurable expiry
- **Password Hashing** — bcrypt with automatic salt generation
- **CORS Middleware** — Configurable cross-origin policies
- **Input Validation** — Pydantic models on all endpoints
- **Health Checks** — Docker HEALTHCHECK + API health endpoint

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with ❤️ using open-source tools. Zero API costs. Enterprise-grade architecture.
