# PonniRAG System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    USER BROWSER                                      │
│                              http://<host>:3000                                      │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER COMPOSE NETWORK                                  │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                         NGINX (frontend container)                              │ │
│  │                              Port 3000 → 80                                     │ │
│  │                                                                                 │ │
│  │   Static Assets (/*)           API Proxy (/api/*, /health)                      │ │
│  │   React SPA bundle  ────────►  proxy_pass http://api:8000                       │ │
│  │   Gzip compression             SSE: proxy_buffering off                         │ │
│  │   try_files → index.html       proxy_read_timeout 300s                          │ │
│  └──────────────┬──────────────────────────────────┬───────────────────────────────┘ │
│                 │                                   │                                 │
│                 │ Static files                      │ /api/* proxied                  │
│                 ▼                                   ▼                                 │
│  ┌──────────────────────────┐    ┌─────────────────────────────────────────────────┐ │
│  │     REACT FRONTEND       │    │              FASTAPI (api container)             │ │
│  │                          │    │                   Port 8000                      │ │
│  │  Pages:                  │    │              Uvicorn, 2 workers                  │ │
│  │  / ─── Home (Search)     │    │                                                 │ │
│  │  /library ─── Volumes    │    │  Endpoints:                                     │ │
│  │  /library/:vol ─ Issues  │    │  POST /api/ask ──────── JSON Q&A                │ │
│  │  /library/:vol/:iss ─ PDF│    │  POST /api/ask/stream ─ SSE streaming           │ │
│  │  /about ─── About        │    │  GET  /api/search ───── Archive search          │ │
│  │                          │    │  GET  /api/authors ──── Author listing           │ │
│  │  Services:               │    │  GET  /api/topics ───── Topic search             │ │
│  │  api.js (axios)          │    │  GET  /api/library ──── Volumes/Issues/PDF       │ │
│  │  translations.js (i18n)  │    │  GET  /api/cache/stats  Cache metrics            │ │
│  │                          │    │  GET  /health ────────── Health check             │ │
│  │  Tech: React 18.2        │    │                                                 │ │
│  │  react-router-dom 6.20   │    │  Tech: FastAPI + Uvicorn                        │ │
│  │  axios 1.6               │    │  Python 3.x, Pydantic                           │ │
│  │  react-markdown 10.1     │    │  sentence-transformers                           │ │
│  └──────────────────────────┘    │  google-genai, pandas                            │ │
│                                  └────────┬──────────┬─────────────────────────────┘ │
│                                           │          │                                │
│                          ┌────────────────┘          └──────────────────┐             │
│                          │                                              │             │
│                          ▼                                              ▼             │
│  ┌──────────────────────────────────────┐          ┌──────────────────────────────┐  │
│  │        QDRANT (qdrant container)     │          │      EXTERNAL SERVICES       │  │
│  │              Port 6333               │          │                              │  │
│  │                                      │          │  ┌────────────────────────┐  │  │
│  │  Collection: tamil_nexus_documents   │          │  │   Gemini 2.5 Flash     │  │  │
│  │                                      │          │  │   (Google Cloud)       │  │  │
│  │  Named Vectors:                      │          │  │                        │  │  │
│  │  ├─ dense: 1024-dim cosine           │          │  │  LLM answer generation │  │  │
│  │  │  (intfloat/multilingual-e5-large) │          │  │  CSV gist summarize    │  │  │
│  │  └─ sparse: BM25-style hashed tokens │          │  │  Streaming responses   │  │  │
│  │                                      │          │  └────────────────────────┘  │  │
│  │  Payload: type, content, chunk_id,   │          │                              │  │
│  │  metadata (doc_id, title, author,    │          │  ┌────────────────────────┐  │  │
│  │  volume, year, issue, s3_key)        │          │  │      AWS S3            │  │  │
│  │                                      │          │  │   Bucket: ponni-dev    │  │  │
│  │  Volume: qdrant_data (persistent)    │          │  │   Region: ap-south-1   │  │  │
│  │  Health: TCP check every 15s         │          │  │                        │  │  │
│  └──────────────────────────────────────┘          │  │  PDFs, DOCX, JSON      │  │  │
│                                                     │  │  Extracted articles    │  │  │
│                                                     │  └────────────────────────┘  │  │
│                                                     └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Query Processing Pipeline

```
                                    User Question
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │  Health Check  │
                                 │   (Qdrant)     │
                                 └───────┬───────┘
                                         │
                                         ▼
                                 ┌───────────────┐     hit
                                 │ Response Cache ├──────────► Return cached result
                                 │  (TTL 1hr,    │
                                 │   LRU 100)    │
                                 └───────┬───────┘
                                         │ miss
                                         ▼
                              ┌─────────────────────┐
                              │   Query Routing      │
                              │                      │
                              │  Pattern match for   │
                              │  author/topic/issue  │
                              │  keywords?           │
                              └──────┬───────┬──────┘
                                     │       │
                            CSV match│       │ No match
                                     │       │
                    ┌────────────────┘       └────────────────┐
                    ▼                                          ▼
        ┌───────────────────────┐               ┌──────────────────────────┐
        │     CSV FLOW          │               │    VECTOR SEARCH FLOW    │
        │                       │               │                          │
        │ EnhancedAuthorQuery   │               │  HybridQdrantSearch      │
        │ System                │               │                          │
        │  • Author listing     │               │  ┌────────────────────┐  │
        │  • Topic → authors    │               │  │  Dense Prefetch    │  │
        │  • Author → topics    │               │  │  E5 embeddings     │  │
        │  • Issue counts       │               │  │  cosine ≥ 0.8      │  │
        │                       │               │  │  limit: 60         │  │
        │ Source: summary.csv   │               │  └────────┬───────────┘  │
        │ (pandas DataFrame)    │               │           │              │
        └───────────┬───────────┘               │  ┌────────┴───────────┐  │
                    │                            │  │     RRF Fusion     │  │
                    │ csv_response               │  │  1/(k + rank)     │  │
                    ▼                            │  └────────┬───────────┘  │
        ┌───────────────────────┐               │  ┌────────┴───────────┐  │
        │  CSV LLM Gist         │               │  │  Sparse Prefetch   │  │
        │                       │               │  │  BM25-style tokens │  │
        │  _CSV_SYSTEM_PROMPT   │               │  │  no threshold      │  │
        │  (50-150 word gist)   │               │  │  limit: 60         │  │
        │                       │               │  └────────────────────┘  │
        │  "Summarize counts,   │               │           │              │
        │   key names, trends.  │               │           ▼              │
        │   Don't repeat data." │               │  Top 30 fused results    │
        │           │           │               └──────────┬───────────────┘
        │           ▼           │                          │
        │  ┌─────────────────┐  │                          ▼
        │  │  Gemini 2.5     │  │               ┌──────────────────────────┐
        │  │  Flash API      │  │               │ merge_consecutive_chunks  │
        │  └────────┬────────┘  │               │                          │
        │           │           │               │ For each unique document: │
        │           ▼           │               │ • Retrieve all chunks     │
        │  _combine_csv_answer  │               │ • Sort by chunk_id        │
        │  ┌─────────────────┐  │               │ • Join into full content  │
        │  │ LLM Gist        │  │               │ • Filter < 50 words       │
        │  │ ─────────────── │  │               │ • Sort by RRF score       │
        │  │ ---             │  │               └──────────┬───────────────┘
        │  │ தரவுத்தள தகவல்: │  │                          │
        │  │ {raw CSV data}  │  │                          ▼
        │  └─────────────────┘  │               ┌──────────────────────────┐
        │                       │               │  _select_relevant_docs    │
        │  sources: []          │               │                          │
        │  (no evidence card)   │               │  Score-gap filtering:     │
        └───────────┬───────────┘               │  • Floor: ≥ 30% of top   │
                    │                            │  • Gap: ≥ 40% of prev    │
                    │                            │  • Min: 1, Max: 10       │
                    │                            │  • Dedup by content hash │
                    │                            └──────────┬───────────────┘
                    │                                        │
                    │                                        ▼
                    │                            ┌──────────────────────────┐
                    │                            │  build_context_from_docs  │
                    │                            │                          │
                    │                            │  15,000 char budget       │
                    │                            │  ÷ N relevant docs        │
                    │                            │  = equal excerpt per doc  │
                    │                            │  (min 500 chars each)     │
                    │                            └──────────┬───────────────┘
                    │                                        │
                    │                                        ▼
                    │                            ┌──────────────────────────┐
                    │                            │  search_csv_semantic      │
                    │                            │  (top 3 matching rows)    │
                    │                            └──────────┬───────────────┘
                    │                                        │
                    │                                        ▼
                    │                            ┌──────────────────────────┐
                    │                            │  Gemini 2.5 Flash API    │
                    │                            │                          │
                    │                            │  TAMIL_ANSWER_SYSTEM_    │
                    │                            │  PROMPT                  │
                    │                            │  (200-500 words,         │
                    │                            │   Ponni context,         │
                    │                            │   formatting rules)      │
                    │                            │                          │
                    │                            │  Fallback:               │
                    │                            │  generate_extractive_    │
                    │                            │  answer() if LLM fails   │
                    │                            └──────────┬───────────────┘
                    │                                        │
                    │                                        ▼
                    │                            ┌──────────────────────────┐
                    │                            │  format_sources           │
                    │                            │                          │
                    │                            │  Same _select_relevant_  │
                    │                            │  docs set → full merged  │
                    │                            │  content as evidence     │
                    │                            └──────────┬───────────────┘
                    │                                        │
                    └────────────────┬───────────────────────┘
                                     │
                                     ▼
                             ┌───────────────┐
                             │ Cache Store    │
                             │ + Return       │
                             │ Response       │
                             └───────────────┘
```

---

## Backend Module Structure (`src/db/`)

```
hybrid_search.py   ← Orchestrator (ask_question, embeddings, caching, model loaders)
       │               Re-exports all public names for backward compatibility
       │
       ├── tamil_text.py      Tamil NLP utilities, fuzzy matching, pattern bank
       │                      (leaf — no internal imports)
       │
       ├── csv_queries.py     CSV query pipeline: author/topic/issue queries,
       │                      EnhancedAuthorQuerySystem, formatting
       │                      (imports tamil_text)
       │
       ├── llm.py             Gemini LLM layer: prompts, sync/async/streaming
       │                      generation, extractive fallback
       │                      (leaf — reads env vars directly)
       │
       └── search.py          Vector search document processing: chunk retrieval,
                              merging, relevance filtering, context building
                              (leaf — constants defined locally)

api.py             FastAPI endpoints (imports from hybrid_search)
qdrant_indexer.py   Vector indexing with E5 prefixes
streamlit_app.py    Legacy Streamlit UI
```

All external consumers (`api.py`, tests) import from `hybrid_search` — the re-export layer ensures backward compatibility.

---

## Data Indexing Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│    AWS S3        │     │  Text Extraction  │     │ Article Separation   │
│                  │     │                   │     │                      │
│  ponni-dev/      │     │  text_extraction  │     │  article_seperation  │
│  Raw_Proof_Read  │────►│  .py              │────►│  .py                 │
│  _Content/       │     │                   │     │                      │
│                  │     │  PDF/DOCX → text  │     │  Pattern matching    │
│  PDFs, DOCX      │     │  OCR fallback     │     │  Boundary detection  │
└──────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                              │
                                                              ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│    Qdrant        │     │  Qdrant Indexer   │     │ Content Extraction   │
│                  │     │                   │     │                      │
│  tamil_nexus_    │     │  qdrant_indexer   │     │  content_extraction  │
│  documents       │◄────│  .py              │◄────│  .py                 │
│                  │     │                   │     │                      │
│  Dense: 1024-dim │     │  Chunk (500 char) │     │  Metadata: title,    │
│  Sparse: BM25    │     │  E5 embed (dense) │     │  author, year, issue │
│  Payload: meta   │     │  BM25 hash (sparse)│    │  → JSON to S3        │
└──────────────────┘     │  Batch upsert 100 │     └──────────────────────┘
                         └──────────────────┘
```

---

## Tech Stack

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| React | 18.2.0 | UI framework |
| react-router-dom | 6.20.0 | Client-side routing |
| axios | 1.6.0 | HTTP client for API calls |
| react-markdown | 10.1.0 | Render LLM markdown responses |
| Nginx | alpine | Production static server + API reverse proxy |
| Node.js | 18 (alpine) | Build toolchain |

### Backend

| Technology | Version | Purpose |
|---|---|---|
| FastAPI | - | REST API framework |
| Uvicorn | - | ASGI server (2 workers) |
| Pydantic | - | Request/response validation |
| Python | 3.x | Runtime |

### ML / AI

| Technology | Purpose |
|---|---|
| intfloat/multilingual-e5-large | Dense embeddings (1024-dim, cosine) |
| sentence-transformers | Embedding model inference |
| PyTorch (CUDA 11.8 / CPU) | Model runtime with GPU/CPU auto-detection |
| Google Gemini 2.5 Flash (google-genai) | LLM answer generation, CSV summarization, streaming |
| scikit-learn | Cosine similarity for CSV semantic search |
| sentencepiece | Tokenization |

### Data & Storage

| Technology | Purpose |
|---|---|
| Qdrant | Vector database — hybrid search (dense + sparse + RRF) |
| AWS S3 (boto3) | Document storage (PDFs, DOCX, extracted JSON) |
| pandas | CSV article metadata (summary.csv) |
| docx2txt | DOCX text extraction |

### Infrastructure

| Technology | Purpose |
|---|---|
| Docker Compose | Multi-container orchestration (3 services) |
| NVIDIA CUDA 12.1.0 | GPU acceleration for embeddings (optional) |
| Docker named volumes | Qdrant data persistence |

---

## Environment Variables

| Variable | Service | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | api | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | api | S3 secret key |
| `AWS_DEFAULT_REGION` | api | S3 region (ap-south-1) |
| `GEMINI_API_KEY` | api | Google Gemini API key |
| `GEMINI_MODEL` | api | LLM model (default: gemini-2.5-flash) |
| `QDRANT_HOST` | api | Qdrant hostname (default: qdrant) |
| `QDRANT_PORT` | api | Qdrant port (default: 6333) |
| `USE_MOCK_DATA` | api | Skip S3/Qdrant, use sample data (true/false) |

---

## Port Map

| Port | Service | Access |
|---|---|---|
| 3000 | Nginx (frontend) | Public — user-facing |
| 8000 | FastAPI (api) | Internal — proxied via Nginx |
| 6333 | Qdrant | Internal — API container only |

---

## Key Configuration Constants

```python
# Collection
COLLECTION_NAME    = "tamil_nexus_documents"
EMBEDDING_MODEL    = "intfloat/multilingual-e5-large"
EMBEDDING_DIM      = 1024

# Search
SCORE_THRESHOLD    = 0.8     # min cosine similarity for dense prefetch
RELEVANCE_FLOOR    = 0.3     # doc score ≥ 30% of top → included
RELEVANCE_GAP      = 0.4     # doc score < 40% of prev → cutoff
MAX_SOURCES        = 10      # max evidence cards
CONTEXT_BUDGET     = 15000   # chars distributed equally across docs

# Indexing
CHUNK_SIZE         = 500     # characters per chunk
BATCH_SIZE         = 100     # upsert batch size

# Cache
CACHE_MAX_SIZE     = 100     # LRU entries
CACHE_TTL          = 3600    # seconds (1 hour)

# LLM
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_TOKENS  = 2048
GEMINI_TOP_P       = 0.9

# S3
BUCKET_NAME        = "ponni-dev"
INPUT_PREFIX       = "Raw_Proof_Read_Content/"
OUTPUT_PREFIX      = "output_json/"
```
