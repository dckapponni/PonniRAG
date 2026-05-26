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
│  │  /library/:vol ─ Issues  │    │  POST /api/ask ──────── JSON Q&A (+ tag filter) │ │
│  │  /library/:vol/:iss ─ PDF│    │  POST /api/ask/stream ─ SSE streaming           │ │
│  │  /tags ─── Tag Browse     │    │  GET  /api/search ───── Archive search (+ tags) │ │
│  │  /about ─── About        │    │  GET  /api/tags ─────── Tag taxonomy + counts     │ │
│  │                          │    │  GET  /api/tags/:id/articles ── Tag articles     │ │
│  │                          │    │  GET  /api/authors ──── Author listing           │ │
│  │  Services:               │    │  GET  /api/topics ───── Topic search             │ │
│  │  api.js (axios)          │    │  GET  /api/library ──── Volumes/Issues/PDF       │ │
│  │  translations.js (i18n)  │    │  GET  /api/cache/stats  Cache metrics            │ │
│  │                          │    │  GET  /health ────────── Health check             │ │
│  │  Tech: React 18.2        │    │                                                 │ │
│  │  react-router-dom 6.20   │    │  Tech: FastAPI + Uvicorn                        │ │
│  │  axios 1.6               │    │  Python 3.x, Pydantic                           │ │
│  │  react-markdown 10.1     │    │  sentence-transformers, cross-encoder            │ │
│  └──────────────────────────┘    │  google-genai, pandas, fastembed                  │ │
│                                  └────────┬──────────┬─────────────────────────────┘ │
│                                           │          │                                │
│                          ┌────────────────┘          └──────────────────┐             │
│                          │                                              │             │
│                          ▼                                              ▼             │
│  ┌──────────────────────────────────────┐          ┌──────────────────────────────┐  │
│  │        QDRANT (qdrant container)     │          │      EXTERNAL SERVICES       │  │
│  │              Port 6333               │          │                              │  │
│  │                                      │          │  ┌────────────────────────┐  │  │
│  │  Collection: qdrant_indexer           │          │  │   Gemini 2.5 Flash     │  │  │
│  │                                      │          │  │   (Google Cloud)       │  │  │
│  │  Named Vectors:                      │          │  │                        │  │  │
│  │  ├─ dense: 1024-dim cosine           │          │  │  LLM answer generation │  │  │
│  │  │  (intfloat/multilingual-e5-large) │          │  │  CSV gist summarize    │  │  │
│  │  └─ sparse: BM25 (Qdrant/bm25        │
│  │     via fastembed)                   │          │  │  Streaming responses   │  │  │
│  │                                      │          │  └────────────────────────┘  │  │
│  │  Payload: type, content, chunk_id,   │          │                              │  │
│  │  metadata (doc_id, title, author,    │          │  ┌────────────────────────┐  │  │
│  │  volume, year, issue, s3_key,        │
│  │  tags, tags_tamil)                   │          │  │      AWS S3            │  │  │
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
                                 │ • Truncate to  │
                                 │   500 chars    │
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
        │  • Issue counts       │               │  │  cosine ≥ 0.65     │  │
        │                       │               │  │  limit: 200        │  │
        │ Source: summary.csv   │               │  └────────┬───────────┘  │
        │ (pandas DataFrame)    │               │           │              │
        └───────────┬───────────┘               │  ┌────────┴───────────┐  │
                    │                            │  │     RRF Fusion     │  │
                    │ csv_response               │  │  1/(k + rank)     │  │
                    ▼                            │  └────────┬───────────┘  │
        ┌───────────────────────┐               │  ┌────────┴───────────┐  │
        │  CSV LLM Gist         │               │  │  Sparse Prefetch   │  │
        │                       │               │  │  BM25 (fastembed)   │  │
        │  _CSV_SYSTEM_PROMPT   │               │  │  no threshold      │  │
        │  (50-150 word gist)   │               │  │  limit: 200        │  │
        │                       │               │  └────────────────────┘  │
        │  "Summarize counts,   │               │           │              │
        │   key names, trends.  │               │           ▼              │
        │   Don't repeat data." │               │  Top fused results           │
        │           │           │               │  (optional tag filter:       │
        │           │           │               │   MatchAny metadata.tags)    │
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
        │                       │               │  Cross-Encoder Reranker   │
        │  sources: []          │               │  (BAAI/bge-reranker-      │
        │  (no evidence card)   │               │   v2-m3)                  │
        └───────────┬───────────┘               │                          │
                    │                            │  • Heading-phrase pinning │
                    │                            │  • Adaptive score floor   │
                    │                            │    (40% of top score)     │
                    │                            │  • Gap cutoff             │
                    │                            │    (50% of prev score)    │
                    │                            │  • Top-in: 30, Out: 5    │
                    │                            │  • Min: 3 results        │
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
                    │                            │  max_output_tokens: 4096 │
                    │                            │  temperature: 0.0        │
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
                    │                            │  Reranked docs → full    │
                    │                            │  merged content as       │
                    │                            │  evidence (incl. tags)   │
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
       │                      generation, extractive fallback, S3 context loading
       │                      (imports guardrails for prompt hardening)
       │
       ├── search.py          Vector search document processing: chunk retrieval,
       │                      merging (grouped by title), relevance filtering,
       │                      context building
       │                      (imports COLLECTION_NAME from embeddings)
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

embeddings.py       Embedding engine: dense (E5) + sparse (BM25) vectors
                     Thread-safe singleton loading, CSV semantic search
                     HybridQdrantSearch class with tag filtering support
                     (imports sparse.py for BM25)

sparse.py           BM25 sparse embeddings via fastembed (Qdrant/bm25 model)
                     Separate doc/query embedding functions with weight modes
                     Controlled by ENABLE_RERANKER flag
                     (leaf — fastembed only)

reranker.py         Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
                     Heading-phrase pinning, adaptive score flooring,
                     clear-win skip optimization
                     Env-configurable: model, top-in/out, thresholds
                     (leaf — sentence-transformers CrossEncoder)

snapshot_manager.py  Qdrant snapshot backup/restore via S3
                     Reindex detection using embedding fingerprint + source hash

update_csv_tags.py   Syncs tags from Qdrant back to summary.csv
                     Matches rows by doc_id + doc_issue + article_no (chunk_id=0)
                     Adds வகை (category) column with Tamil tag names

api.py              FastAPI endpoints (imports from hybrid_search)
                     S3 image proxy for volume/issue covers
                     Tag endpoints: /api/tags, /api/tags/{id}/articles
                     Tag filtering on /api/ask and /api/search

qdrant_indexer.py    Vector indexing with E5 passage:/query: prefixes
                     Integrates ArticleTagger during indexing (2-pass)
                     Stores tags + tags_tamil in Qdrant payload

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
│  qdrant_indexer  │     │  qdrant_indexer   │     │  content_extraction  │
│  (collection)    │◄────│  .py              │◄────│  .py                 │
│                  │     │                   │     │                      │
│  Dense: 1024-dim │     │  Pass 1: Collect  │     │  Metadata: title,    │
│  Sparse: BM25    │     │    articles for   │     │  author, year, issue │
│  Payload: meta   │     │    tagger training│     │  → JSON to S3        │
│  + tags          │     │  Pass 2: Tag +    │     └──────────────────────┘
│                  │     │    chunk + embed   │
│  Snapshots → S3  │     │    + batch upsert  │
└──────────────────┘     └──────────────────┘
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
│                  │     │  2. TF-IDF fallback:  │     │    (tags +       │
│                  │     │     trained on labeled │     │     tags_tamil)  │
│                  │     │     articles           │     │  • summary.csv   │
│                  │     │     threshold: 0.15    │     │    (வகை column)  │
└─────────────────┘     │  3. Serial detection: │     └──────────────────┘
                         │     multi-part fiction │
                         └─────────────────────┘

TAXONOMY (15 categories):
  FICTION, EDITORIAL, LITERARY_REVIEW, POETRY, QA_COLUMN,
  ARTS_CULTURE, SATIRE_HUMOR, CLASSICAL_LIT, PUBLIC_FORUM,
  WOMENS_ISSUES, RELIGIOUS_DEBATE, CHILDRENS, POLITICAL,
  BIOGRAPHY, GENERAL

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

## Cross-Encoder Reranking Pipeline (Phase 1.1)

```
Fused RRF Results (top 30)
       │
       ▼
┌──────────────────────────┐
│  Heading-Phrase Pinning   │
│                           │
│  If query phrase matches  │
│  article title exactly:   │
│  • Pin to top (skip CE)   │
│  • Min 2-word phrase      │
│  • Max 3 pinned docs      │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Cross-Encoder Scoring    │
│                           │
│  Model: BAAI/bge-reranker │
│         -v2-m3            │
│  Input: (query, doc) pairs│
│  Batch size: 16           │
│  Max chars/doc: 800       │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Adaptive Score Cutoff    │
│                           │
│  Floor: score ≥ 40% of   │
│         top_score         │
│  Gap:   score ≥ 50% of   │
│         prev_score        │
│  Min output: 3            │
│  Max output: 5            │
└──────────┬───────────────┘
           │
           ▼
  Reranked top-K results
```

All thresholds env-configurable. Set `ENABLE_RERANKER=0` to bypass entirely.

---

## Qdrant Snapshot Management (Phase 1.1)

```
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  snapshot_manager │     │  Reindex Detection   │     │  S3 Backup       │
│  .py              │     │                      │     │                  │
│                   │────►│  SHA-256 fingerprint  │     │  ponni-dev/      │
│  On startup:      │     │  of: model + dim +    │     │  snapshots/      │
│  • Check if index │     │  chunk_size           │     │                  │
│    is current     │     │                      │     │  Save/restore    │
│  • Restore from   │     │  SHA-256 hash of     │     │  Qdrant snapshots│
│    S3 if needed   │     │  S3 source file keys  │     │                  │
│                   │     │  + timestamps         │     │                  │
└──────────────────┘     └─────────────────────┘     └──────────────────┘
```

---

## Frontend: Tag Browse Page (`/tags`) (Phase 1.1)

```
┌─────────────────────────────────────────────────────────────────┐
│  TagBrowse.js                                                    │
│                                                                  │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  LEFT SIDEBAR        │  │  RIGHT CONTENT PANE              │  │
│  │                      │  │                                  │  │
│  │  Volume Accordion    │  │  State 1: Empty                  │  │
│  │  ├─ Volume 1 ▼       │  │    "Select a volume to browse"   │  │
│  │  │  ├─ Issue 6       │  │                                  │  │
│  │  │  ├─ Issue 7       │  │  State 2: Issue Articles         │  │
│  │  │  └─ PONGAL        │  │    Article list with:            │  │
│  │  ├─ Volume 2 ▶       │  │    • Title, author, year         │  │
│  │  └─ Volume 3 ▶       │  │    • Tag badges                  │  │
│  │                      │  │    • Category/search filters     │  │
│  │  ─────────────────   │  │    • PDF link button             │  │
│  │  Category Filter ▼    │  │                                  │  │
│  │  (from /api/tags)    │  │  State 3: Article Detail          │  │
│  │                      │  │    Full content, metadata,        │  │
│  │  Search Input        │  │    tags, word count, stats        │  │
│  │  (client-side filter)│  │                                  │  │
│  │                      │  │                                  │  │
│  │  Result Count        │  │                                  │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
│                                                                  │
│  API Calls:                                                      │
│  • getTags() — taxonomy load                                    │
│  • getVolumes() — volume list                                   │
│  • getVolumeIssues(volId) — lazy-load on expand                 │
│  • getIssueArticles(volId, issId) — articles for issue          │
│  • getArticleContent(docId, docIssue, articleNo) — full article │
│  • getPDFLink(volId, issId) — PDF URL                           │
└─────────────────────────────────────────────────────────────────┘
```

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
       │                              • 15,000 char max per message
       ▼
Last 3 Q&A turns (6 messages)
sent as `history` in POST body
       │
       ▼                              Request ID tracking prevents
Gemini receives conversation context  stale stream callbacks
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
| BAAI/bge-reranker-v2-m3 | Cross-encoder reranking (multilingual) |
| Qdrant/bm25 (fastembed) | BM25 sparse embeddings |
| PyTorch (CUDA 11.8 / CPU) | Model runtime with GPU/CPU auto-detection |
| Google Gemini 2.5 Flash (google-genai) | LLM answer generation, CSV summarization, streaming |
| scikit-learn | Cosine similarity for CSV semantic search, TF-IDF for tagging |
| sentencepiece | Tokenization |

### Data & Storage

| Technology | Purpose |
|---|---|
| Qdrant | Vector database — hybrid search (dense + sparse + RRF) + tag filtering |
| AWS S3 (boto3) | Document storage (PDFs, DOCX, extracted JSON, Qdrant snapshots) |
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
| `ENABLE_RERANKER` | api | Enable cross-encoder reranking (default: 1) |
| `RERANKER_MODEL` | api | Reranker model (default: BAAI/bge-reranker-v2-m3) |
| `RERANKER_TOP_IN` | api | Candidates fed to reranker (default: 30) |
| `RERANKER_TOP_OUT` | api | Results kept after reranking (default: 5) |
| `RERANKER_MIN_OUT` | api | Minimum results to return (default: 3) |
| `RERANKER_SCORE_FLOOR_RATIO` | api | Min score as ratio of top (default: 0.4) |
| `RERANKER_GAP_RATIO` | api | Gap cutoff ratio vs prev (default: 0.5) |
| `RERANKER_MAX_CHARS` | api | Per-doc truncation for reranker (default: 800) |
| `RERANKER_BATCH` | api | Reranker batch size (default: 16) |
| `RERANKER_PIN_HEADING_MATCHES` | api | Enable heading-phrase pinning (default: 1) |
| `RERANKER_SKIP_CE_ON_CLEAR_WIN` | api | Skip CE on clear title match (default: 1) |
| `RERANKER_CLEAR_WIN_MIN_PHRASE` | api | Min phrase words for clear win (default: 2) |
| `RERANKER_CLEAR_WIN_MAX_MATCHES` | api | Max pinned docs (default: 3) |
| `RAG_DEBUG` | api | Debug logging flag (default: 0) |

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
COLLECTION_NAME    = "qdrant_indexer"
EMBEDDING_MODEL    = "intfloat/multilingual-e5-large"
EMBEDDING_DIM      = 1024

# Search
SCORE_THRESHOLD    = 0.65    # min cosine similarity for dense prefetch
MAX_SOURCES        = 100     # max evidence cards
CONTEXT_BUDGET     = 15000   # chars distributed equally across docs
MAX_QUERY_LENGTH   = 500     # query truncation limit

# Reranker (Phase 1.1 — env-configurable, defaults shown)
RERANKER_MODEL            = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_IN           = 30      # candidates into reranker
RERANKER_TOP_OUT          = 5       # results out of reranker
RERANKER_MIN_OUT          = 3       # minimum results guaranteed
RERANKER_SCORE_FLOOR_RATIO = 0.4   # score ≥ 40% of top
RERANKER_GAP_RATIO        = 0.5    # score ≥ 50% of prev
RERANKER_MAX_CHARS        = 800    # per-doc content truncation

# Tagging (Phase 1.1)
TFIDF_THRESHOLD    = 0.15   # min TF-IDF similarity for category assignment
TAXONOMY_SIZE      = 15     # number of article categories

# Indexing
CHUNK_SIZE         = 500     # characters per chunk
BATCH_SIZE         = 100     # upsert batch size

# Cache
CACHE_MAX_SIZE     = 100     # LRU entries
CACHE_TTL          = 3600    # seconds (1 hour)

# LLM
GEMINI_MODEL       = "gemini-2.5-flash"  # configurable via env
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_TOKENS  = 4096
GEMINI_TOP_P       = 0.9

# S3
BUCKET_NAME        = "ponni-dev"
INPUT_PREFIX       = "Raw_Proof_Read_Content/"
OUTPUT_PREFIX      = "output_json/"
```

---

## API Pydantic Models (Phase 1.1 additions)

```python
# Tag endpoints
TagInfo           = {id, tamil, english, count}
TagsResponse      = {success, tags: List[TagInfo]}
TagArticleInfo    = {doc_id, doc_issue, title, author_name, year, tags}
TagArticlesResponse = {success, tag_id, tag_tamil, count, articles}

# Enhanced existing models
QuestionRequest.tags: Optional[List[str]]       # filter by tag IDs
SourceDocument.tags:  Optional[List[str]]       # tags on evidence cards
ArticleContentResponse.tags: Optional[List[str]] # tags on full article
```
