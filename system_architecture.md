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
│  │  /tags ─── Categories    │    │  GET  /api/search ───── Archive search          │ │
│  │  /about ─── About        │    │  GET  /api/tags ─────── Tag categories + counts  │ │
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
                                 │  Input Guard   │
                                 │  (guardrails)  │
                                 │                │
                                 │ • Strip quotes │
                                 │ • Sanitize     │
                                 │   control chars│
                                 │ • NFC normalize│
                                 │ • Injection    │
                                 │   detection    │
                                 └───────┬───────┘
                                         │ (blocked if HIGH injection)
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
       ├── guardrails.py      Prompt security: injection detection, query sanitization,
       │                      history validation, output leakage checks
       │                      (leaf — stdlib only, no project imports)
       │
       ├── tamil_text.py      Tamil NLP utilities, fuzzy matching, pattern bank
       │                      (leaf — no internal imports)
       │
       ├── csv_queries.py     CSV query pipeline: author/topic/issue queries,
       │                      EnhancedAuthorQuerySystem, author display formatting
       │                      (imports tamil_text)
       │
       ├── llm.py             Gemini LLM layer: prompts, sync/async/streaming
       │                      generation, extractive fallback
       │                      (imports guardrails for prompt hardening)
       │
       ├── search.py          Vector search document processing: chunk retrieval,
       │                      merging, relevance filtering, context building
       │                      (leaf — constants defined locally)
       │
       ├── cache.py           Thread-safe LRU response cache with TTL
       │                      Auto-invalidates when Qdrant points_count changes
       │                      (leaf — stdlib only)
       │
       └── retry.py           Exponential backoff with jitter for Qdrant/Gemini
                              Sync + async retry loops, service-specific predicates
                              (leaf — stdlib only)

article_tagger.py   Hybrid article tagger: rule-based + TF-IDF fallback
                     TAXONOMY (15 categories), ArticleTagger class
                     Fully offline, deterministic, no LLM

embeddings.py       Embedding engine: dense (E5) + sparse (BM25-style) vectors
                     Thread-safe singleton loading, CSV semantic search
                     HybridQdrantSearch class with tag filtering

snapshot_manager.py  Qdrant snapshot backup/restore via S3
                     Reindex detection using embedding fingerprint + source hash

api.py              FastAPI endpoints (imports from hybrid_search)
                     S3 image proxy for volume/issue covers
                     Includes /api/tags and /api/tags/{id}/articles
qdrant_indexer.py    Vector indexing with E5 passage:/query: prefixes
streamlit_app.py     Legacy Streamlit UI (Ask AI, Library, Tags/Categories, About)
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

## Security — Guardrails Pipeline (`guardrails.py`)

```
User Input
    │
    ├─► Strip control characters (keep \n, \t, space)
    ├─► Strip quotation marks (ASCII + smart/curly quotes + guillemets)
    ├─► Neutralize separators (====, ```)
    ├─► Collapse excessive whitespace
    ├─► NFC Unicode normalization
    │
    ▼
Injection Detection
    │
    ├─► HIGH patterns (block): instruction override, role switching,
    │   system prompt extraction, delimiter injection
    ├─► MEDIUM patterns (warn): "from now on", env var requests
    │
    ▼
History Validation (for multi-turn conversations)
    │
    ├─► Enforce user/assistant alternating pattern
    ├─► Truncate content at 5000 chars/turn, max 20 turns
    ├─► Drop user turns with HIGH injection signals
    │
    ▼
Output Guardrails
    │
    ├─► Check for leakage of API keys, bucket names, model names,
    │   prompt variable names, source file names
    └─► Replace leaked responses with safe Tamil refusal message
```

---

## Article Tagging Pipeline

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Article Content │     │  article_tagger.py   │     │  15 Categories   │
│  (title + body)  │────►│                      │────►│  (1–3 tags each) │
│                  │     │  1. Rule-based:      │     │                  │
│  From Qdrant     │     │     keyword patterns  │     │  Written to:     │
│  payload or CSV  │     │     per category      │     │  • Qdrant payload│
│                  │     │  2. TF-IDF fallback:  │     │  • summary.csv   │
│                  │     │     trained on labeled │     │    (வகை column)  │
│                  │     │     articles           │     │                  │
└─────────────────┘     │  3. Serial detection: │     └──────────────────┘
                         │     multi-part fiction │
                         └─────────────────────┘

update_csv_tags.py — Syncs tags from Qdrant back to summary.csv
                     Matches rows by doc_id + doc_issue + article_no (chunk_id=0)
```

---

## S3 Image Proxy

```
Browser                      FastAPI                         AWS S3
  │                             │                               │
  │  GET /api/images/          │                               │
  │  volumes/3/cover           │  _volume_cover_s3_key(3)      │
  │ ───────────────────────►   │  → "Front_cover_of_volumes/   │
  │                             │     Volumes/Volume3.png"      │
  │                             │  ────────────────────────►    │
  │                             │                  ◄────────    │
  │  ◄─────────────────────    │  Response(image/png)          │
  │  Cache-Control: 1hr        │  fallback: .jpg if .png fails │
  │                             │                               │
  │  GET /api/images/          │                               │
  │  volumes/3/issues/7/cover  │  _issue_cover_s3_key(3, 7)   │
  │ ───────────────────────►   │  → "Front_cover_of_volumes/   │
  │                             │     volume 3 cover images/    │
  │                             │     VOL3 - 7 - 1949.jpg"     │
  │                             │  ────────────────────────►    │
```

No presigned URLs — proxy endpoints never expire. Browser caches for 1 hour.
PDF links use Google Drive URLs directly from `magazine_registry.json`.

---

## Magazine Registry (`config/magazine_registry.json`)

Single source of truth for all magazine metadata:

```json
{
  "name": "ponni",
  "s3": { "bucket": "ponni-dev", "region": "ap-south-1", ... },
  "volumes": [
    {
      "id": 1,
      "issues": [
        { "num": "6", "year": 1947, "pdf_url": "https://drive.google.com/..." },
        { "num": "7", "year": 1947, "pdf_url": "..." },
        { "num": "PONGAL", "year": 1948, "pdf_url": "..." }
      ]
    }
  ]
}
```

- Adding new issues only requires editing this file
- `config.py` loads the registry via `get_magazine_config("ponni")`
- Issue numbers are strings (supports "PONGAL" special issues)
- Volumes can span multiple years (year range derived from per-issue years)

---

## Conversation History (Multi-Turn)

```
Frontend (Home.js)                    Backend (hybrid_search.py)

messages state:                       HistoryMessage validation:
[                                     • role: "user" | "assistant"
  {role: "user", content: "Q1"},      • content: min 1, max 5000 chars
  {role: "assistant", content: "A1"}, • Max 20 turns
  {role: "user", content: "Q2"},      • Alternating role pattern enforced
  {role: "assistant", content: "A2"}, • HIGH injection → turn dropped
]                                     • Content sanitized per turn
       │
       ▼
Last 3 Q&A turns (6 messages)
sent as `history` in POST body
       │
       ▼
Gemini receives conversation context
→ enables follow-up questions
```

Cache is skipped when conversation history is present (each turn is unique).

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
| Qdrant snapshots + S3 | Index backup/restore (`snapshot_manager.py`) |
| Exponential backoff | Retry with jitter for Qdrant/Gemini (`retry.py`) |

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
