# Architecture Decision Record (ADR) — PonniRAG

**Date:** 2026-02-23
**Status:** Accepted
**Project:** PonniRAG — RAG system for Tamil literary documents (Ponni magazine, 1947–1955)

## Summary

PonniRAG is a Retrieval-Augmented Generation system built to search, retrieve, and answer questions about the Ponni literary magazine archive. The system uses hybrid search (dense + sparse vectors with RRF fusion) over a Qdrant vector database, with `intfloat/multilingual-e5-large` embeddings (1024-dim, cosine) and cross-encoder reranking (`BAAI/bge-reranker-v2-m3`). The LLM layer — after a failed attempt with a self-hosted Ollama/Tamil-Llama 7B — runs on Google Gemini 2.5 Flash via API. Phase 1.1 added an article tagging system (15 categories, rule-based + TF-IDF), a Tag Browse page, BM25 sparse embeddings via fastembed, and Qdrant snapshot management. The frontend is a React 18 SPA served by Nginx, backed by a FastAPI REST API with SSE streaming, all orchestrated via Docker Compose (3 services).

---

## Timeline of Architectural Evolution

### Phase 1: Project Genesis (Nov–Dec 2025)

| Commit | Date | Milestone |
|--------|------|-----------|
| `ff67898` | 2025-11-20 | Initial commit, README, Tamil language models reference guide |
| `5268ede` | 2025-11-28 | Sample RAG pipeline prototype |
| `aabf4ae`–`372e437` | 2025-12-08–12-10 | Code modularization, extraction pattern improvements |
| `c773316`–`fff0b9e` | 2025-12-11 | S3 data pipeline connected (ponni-dev bucket, ap-south-1) |
| `c24ce21` | 2025-12-14 | First Streamlit UI for Tamil Nexus RAG |
| `691ece8`–`05ca55b` | 2025-12-16 | Streamlit app for RAG, embedding setup, article splitting logic |

**Key decisions:** S3 as document storage, Python data extraction pipeline, first Streamlit prototype.

### Phase 2: Qdrant & Hybrid Search (Dec 2025–Jan 2026)

| Commit | Date | Milestone |
|--------|------|-----------|
| `b6da90e`–`c2cc805` | 2025-12-22 | Hybrid search module and QdrantHybridIndexer implemented |
| `410fc8e`–`ead70ae` | 2025-12-22 | Pre-commit hooks (Black, isort, flake8, gitleaks) |
| `f518819` | 2026-01-04 | Hybrid search integrated |
| `44ff237` | 2026-01-04 | Test cases added |
| `dbc870e`–`d7a3a67` | 2026-01-07 | UI changes, search updates |

**Key decisions:** Qdrant as vector DB, hybrid dense+sparse search with RRF fusion, E5 embeddings, 500-char sentence-aware chunking.

### Phase 3: Streamlit UI Enhancement (Jan 2026)

| Commit | Date | Milestone |
|--------|------|-----------|
| `fb11868` | 2026-01-13 | Streamlit UI overhaul, Qdrant issue fixes |

**Key decisions:** Library/PDF viewer, enhanced search UI. Streamlit reached its limits as a production UI during this phase.

### Phase 4: React Frontend + FastAPI (Feb 2026)

| Commit | Date | Milestone |
|--------|------|-----------|
| `6da6acd`–`295ff79` | 2026-02-02 | Frontend connected to backend |
| `2328e83` | 2026-02-04 | Library section image format fixes |
| `64acc62` | 2026-02-04 | Ollama added for LLM |
| `044413d` | 2026-02-09 | Frontend/data_extraction branches merged |

**Key decisions:** Decoupled architecture — React SPA + FastAPI REST API. Nginx reverse proxy. Docker Compose orchestration.

### Phase 5: Ollama + Tamil-Llama 7B (Feb 2026) — *Failed Experiment*

| Commit | Date | Milestone |
|--------|------|-----------|
| `8476cb6` | 2026-02-10 | Ollama optimization, threshold tuning |
| `4531d7d` | 2026-02-10 | Multi-user conversation support |
| `b04a5e6`–`d7234d3` | 2026-02-11 | Ollama GPU config fix, timing logs |
| `77cde65` | 2026-02-13 | Inference optimization and streaming added |

**Critical bug discovered:** `num_gpu: 1` in Modelfile — only 1 of 32 transformer layers on GPU, making the model CPU-bound with 55–75s response times. Even after fix (`num_gpu: 999`), responses were 15–20s with 10-minute model eviction.

### Phase 6: Gemini 2.5 Flash Migration (Feb 20, 2026)

| Commit | Date | Milestone |
|--------|------|-----------|
| `ebca9a8` | 2026-02-20 | Strict prompt rules, repeat_last_n |
| `45dc183` | 2026-02-20 | LLM switched to Gemini, workflow changes |

**Key decision:** Abandoned self-hosted Ollama for Gemini 2.5 Flash API. Response time dropped from 15–20s to 2–5s. Infrastructure cost eliminated.

### Phase 7: Prompt Optimization & Caching (Feb 2026)

| Commit | Date | Milestone |
|--------|------|-----------|
| `50b19fe`–`d046058` | 2026-02-21 | Context, evidence, CSV evidence fixes |
| `4289c8d`–`4fdfb62` | 2026-02-21 | Dynamic thresholds, random hash fix, tighter thresholds |
| `26a81cc` | 2026-02-21 | LLM gist, context/question repetition fix, prompt optimization |
| `012cfdb` | 2026-02-21 | Document excerpts from top evidence |
| `b0854c7` | 2026-02-22 | CSV prompt expansion, question type expansion |

**Key decisions:** Dual-prompt system (CSV vs vector), response caching (TTL+LRU), SSE streaming, question-type-aware closings.

### Phase 8: Code Modularization (Feb 22, 2026)

| Commit | Date | Milestone |
|--------|------|-----------|
| `f24993d` | 2026-02-22 | Modularization — `hybrid_search.py` split into 5 sub-modules |
| `a503dee` | 2026-02-22 | README update, system architecture document |

**Key decision:** Monolithic `hybrid_search.py` split into `tamil_text.py`, `csv_queries.py`, `llm.py`, `search.py`, and `cache.py`. Orchestrator re-exports all names for backward compatibility.

### Phase 1.1: Tagging, Reranking & Search Quality (Feb–May 2026)

| Commit | Date | Milestone |
|--------|------|-----------|
| `441f0b8`–`03ba199` | 2026-02-23 | Article tagging backend + UI scaffolding for tags |
| `c948140`–`d597ca0` | 2026-02-25–02-26 | Streamlit category page iterations (v1–v6) |
| `f013b9f`–`f340a5b` | 2026-02-27 | React Tag Browse page, conversation history, code modularization |
| `98c3217` | 2026-02-27 | Qdrant snapshot management, removed local index path |
| `be448b5`–`2a85162` | 2026-02-27–02-28 | Guardrails, retry logic, cache persistence, Gemini health check |
| `4f4f5bd`–`07daeb7` | 2026-03-02–03-03 | Bug fixes: Unicode, translations, evidence display, long queries |
| `6534195` | 2026-03-04 | Special characters check in user queries |
| `1c3838c`–`4c8f852` | 2026-03-10–03-11 | S3 image proxy, data standardization, cover image rendering |
| `abfcf74` | 2026-04-29 | Magazine registry PDF URL corrections |
| `d11dc4f`–`b8a4f8d` | 2026-05-05 | New volume data (6, 7, 8), updated article separation logic |
| `1033263`–`fdc3ebc` | 2026-05-06–05-07 | In-app PDF rendering via Google Drive embed |
| `f1285c8`–`82d2a7d` | 2026-05-08–05-09 | Relevance ranking fix, S3 upload utility, PDF rendering fixes |
| `a086c6c`–`67021c8` | 2026-05-11 | Cross-encoder reranker (BAAI/bge-reranker-v2-m3), BM25 via fastembed |
| `aaf3691`–`b2abd6b` | 2026-05-12 | Reranker fixes, model pre-download in Docker, evidence streaming fix |
| `7f86d13` | 2026-05-13 | Reranker top-N tuning (top_in=50) |
| `b0d5bb1` | 2026-05-14 | Docstrings for all modules, API documentation |
| `4add961`–`3450c30` | 2026-05-25 | CSV topic handling fix, table-of-contents queries, DCKAP feedback fixes |

**Key decisions:** Article tagging (15 categories, rule-based + TF-IDF, no LLM). Cross-encoder reranking replaces score-gap filtering. BM25 sparse via fastembed replaces MD5 hash approach. Qdrant snapshots to S3 for index persistence. Tag Browse page with volume/issue/article navigation. Dense score threshold lowered 0.8→0.65, search limit increased 60→200.

---

## Decision Records

### ADR-001: Embedding Model

**Status:** Accepted

**Context:** PonniRAG needs a multilingual embedding model with strong Tamil support for semantic search over literary documents from 1947–1955. The model must handle Tamil Unicode (U+0B80–U+0BFF), literary vocabulary, and agglutinative morphology. It also needs to support asymmetric retrieval (indexing documents vs querying them).

Six embedding models were evaluated on 10 Tamil questions from literary and editorial documents, scored by human reviewers for relevance and accuracy:

| Model | Mean Score | Min | Max | Rating |
|-------|-----------|-----|-----|--------|
| **Multilingual-E5-large** | **82.3** | **79** | **85** | **Selected Model** |
| IndicBERT | 56.8 | 1 | 88 | Good Performer |
| BGE-M3 | 62.3 | 1 | 95 | High Score / Low Relevance |
| Cohere Embed Multilingual | 41.0 | 1 | 67 | Below Average |
| LaBSE | 35.8 | 1 | 57 | Weak Performer |
| Vyakyarth-1-Indic | 32.0 | 1 | 63 | Weak Performer |

Key finding: BGE-M3 had the highest raw similarity scores (up to 95) but all 10 retrieved answers were labeled "Not Relevant" by human reviewers — high cosine similarity does not guarantee contextual relevance. IndicBERT showed inconsistent quality across diverse query types (score range 1–88). Cohere Embed Multilingual retrieved over-descriptive passages. LaBSE and Vyakyarth-1-Indic were weak performers overall.

Multilingual-E5-large scored 82.3 mean across all 10 queries with a tight range of 79–85, and all 10 retrieved answers were assessed as relevant and accurate. Score distribution: 30% scored 85+, 60% scored 80–84, 10% scored 75–79, 0% below 75.

**Decision:** We will use `intfloat/multilingual-e5-large` (1024-dim, cosine) via SentenceTransformers, with E5 `passage:` prefix for indexing and `query:` prefix for search.

**Consequences:** Retrieval quality for Tamil is high — the XLM-R backbone was trained on Tamil Wikipedia, and the 1024-dimensional space provides fine-grained similarity. The tight score range (79–85) demonstrates stable, predictable retrieval with no catastrophic mismatches, unlike BGE-M3 which had wide variance (1–95). Retrieved snippets are concise and directly address the query — correctly identifying authors, causes, and social commentary across diverse query types (education, authorship, social issues, language policy) — without the over-descriptive passages seen with Cohere Embed Multilingual. The same model is reused for evaluation metrics (semantic similarity, BERTScore), eliminating an extra model download. However, the model has a ~2.3 GB RAM footprint and runs on CPU (not on GPU alongside the LLM), which means embedding latency is higher than a smaller model. This is acceptable for a single-user application. The model is free and open-source, so there is no API cost. A limitation is that E5-large is not designed specifically for Indic languages — performance may vary on more regionally specific Tamil content. A larger evaluation set (beyond 10 questions) is recommended for full production validation.

---

### ADR-002: Vector Database

**Status:** Accepted

**Context:** The system requires a vector database that supports both dense and sparse vector search to enable hybrid retrieval for Tamil documents. Self-hosting is preferred to avoid per-query API costs and vendor lock-in. Alternatives considered: Pinecone (managed, per-query cost at $0.08/1M reads on Starter, no self-hosted option), Weaviate (hybrid search support but heavier resource footprint), ChromaDB (simple Python API but no native sparse vector support).

**Decision:** We will use self-hosted Qdrant via Docker, with named vectors (`dense` and `sparse`) in a single collection, and native `Fusion.RRF` for hybrid queries.

**Consequences:** Hybrid search is a single API call — no separate index management or external fusion logic. Docker volume (`qdrant_data`) provides persistence across restarts. Health checks use TCP on port 6333 (15s interval, 60s start period). No API keys or external accounts are needed. The trade-off is self-managed infrastructure, but Docker Compose makes this trivial. Storage cost is ~$2.40/month for an EBS gp3 volume (30 GB) if deployed on EC2 for persistence across spot interruptions.

---

### ADR-003: Search Strategy

**Status:** Accepted (updated in Phase 1.1)

**Context:** Tamil text benefits from both semantic similarity and exact keyword matching due to agglutinative morphology and literary vocabulary. Dense-only search misses exact keywords and rare proper nouns. Sparse-only (BM25) search misses synonyms and paraphrases. A hybrid approach combining both was needed.

**Decision:** We will use two-stage hybrid search with Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking (Phase 1.1).

The search pipeline:
```
Prefetch 1: Dense (E5 embeddings, cosine ≥ 0.65, limit 200)
Prefetch 2: Sparse (BM25 via fastembed Qdrant/bm25, no threshold, limit 200)
Fusion:     RRF — 1/(k + rank)
Reranking:  Cross-encoder (BAAI/bge-reranker-v2-m3), top-in 30 → top-out 5
Optional:   Tag filter (MatchAny on metadata.tags)
```

**Phase 1 (original):** Post-fusion relevance filtering used a score-gap algorithm: absolute floor at 35% of top document score, gap detection at 40% of previous document. Sparse embeddings used deterministic MD5 token hashing (not Python's `hash()`). Dense threshold was 0.8, limit 60.

**Phase 1.1 (current):** Score-gap filtering replaced by cross-encoder reranking (see ADR-013). BM25 sparse embeddings now use fastembed (`Qdrant/bm25` model) with proper IDF weighting instead of MD5 hash approach (see ADR-014). Dense threshold lowered to 0.65 and limit raised to 200 to feed more candidates into the reranker. Optional tag-based filtering allows restricting search to specific article categories.

**Consequences:** Recall is significantly better for Tamil proper nouns and literary terms compared to dense-only search. The cross-encoder reranker provides much higher precision than the score-gap heuristic — it understands query-document semantic relationships rather than relying on embedding distance alone. The trade-off is additional latency from the cross-encoder pass (~200ms for 30 candidates), but heading-phrase pinning and clear-win skip optimization reduce unnecessary CE calls. No additional API cost — reranker runs locally.

---

### ADR-004: LLM — Ollama/Tamil-Llama to Gemini Migration

**Status:** Accepted (supersedes the rejected Ollama approach)

**Context:** The system needed an LLM to generate natural Tamil answers from retrieved document context.

**Phase 1 — Model selection:** Six LLMs were evaluated on the same 10 Tamil questions from literary and editorial documents, with human review of quality, relevance, and coherence:

| Model | Quality | Relevance | Coherence | Key Issues | Rating |
|-------|---------|-----------|-----------|------------|--------|
| **Tamil LLaMA** | **High** | **High** | **High** | Minor factual gaps | **Selected Model** |
| Qwen | Medium | Partial | Medium | Symbol artifacts, repetition | Below Average |
| Sarvam | Low | Low | Low | Incomplete, generic answers | Below Average |
| Meta (LLaMA) | Low | Low | Low | Repeats questions, Hindi code-switching | Poor Performer |
| Krutrim-Spectre-V2 | Low | Low | Low | Inaccurate, misaligned with source | Poor Performer |
| Navarasa | Very Low | None | None | Random fragments, nonsense output | Unusable |

Tamil LLaMA was initially selected because all 10 generated answers were assessed as relevant, it produced grammatically correct fluent Tamil without code-switching to Hindi or English, and it handled both factual and analytical questions effectively.

**Phase 2 — Self-hosted deployment failed:** Tamil LLaMA was deployed via Ollama on a g4dn.xlarge EC2 instance (T4 GPU, 16 GB VRAM). This failed for multiple reasons:

1. **Critical bug — `num_gpu: 1`:** The original Modelfile offloaded only 1 of 32 transformer layers to the GPU. The model ran almost entirely on CPU, producing 55–75s response times.

2. **Even after the fix (`num_gpu: 999`):** Responses were 15–20s (prompt eval ~3–5s, token generation ~10–15s) — still too slow for interactive use.

3. **Model eviction:** Ollama's default 10-minute `keep_alive` evicted the model from GPU VRAM after inactivity, causing 15–30s cold-start reloads.

4. **Infrastructure burden:** Docker container management, CUDA driver compatibility, GGUF model downloads, no easy scaling path.

5. **Cost:** g4dn.xlarge spot at ~$0.16–0.20/hr (~$120–146/mo), on-demand at $0.526/hr (~$384/mo).

Ollama Modelfile (final state):
```
FROM /models/tamil-llama/tamil-llama-7b-v0.1-q8_0.gguf
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
PARAMETER num_gpu 999      # Was initially 1 — the root cause
PARAMETER num_thread 4
PARAMETER temperature 0.0
PARAMETER top_p 0.9
```

**Phase 3 — Comparative testing exposed quality gaps:** Further head-to-head testing of Tamil LLaMA vs Gemini 2.5 Flash on production queries revealed that Tamil LLaMA's initial evaluation overstated its suitability:

| Criteria | Tamil LLaMA | Gemini 2.5 Flash |
|----------|-------------|------------------|
| Tamil language fluency | Good | Good |
| Answer conciseness | High (short, direct) | Medium (detailed) |
| Metadata accuracy (author/year/issue) | Adequate | Accurate |
| Handling "not found" queries | Direct but minimal | Explains with context |
| Article type classification | Weak (misclassified) | Strong |
| Title disambiguation | Not tested | Strong |
| Hallucination risk | Low | Medium |
| Multi-record query handling | Incomplete | Complete |

Query-by-query results:
- "நாரா.நாச்சியப்பன் படைப்புகள் பட்டியல்" — Tamil LLaMA listed 31 works but misclassified all as "செய்திப்பாட்டு"; Gemini correctly listed all 31 with year and issue numbers.
- "காதில் பிறந்த கதை எழுதியவர்கள் யார்?" — Tamil LLaMA returned no answer; Gemini correctly identified authors and disambiguated a similar title.
- "மலர் 2 இதழ் 7 — வள்ளுவர் விருந்து எழுதியவர் யார்?" — Tamil LLaMA returned no answer; Gemini correctly identified the author and differentiated from a related column in other issues.
- "பொன்னி இதழில் முடியரசனின் பங்கு" — Tamil LLaMA hallucinated about a different author ("அழகிரி"); Gemini acknowledged the name was not in the primary contributor list and provided what data was available.

**Decision:** We will migrate to Google Gemini 2.5 Flash API, accessed via the `google-genai` Python SDK. The Ollama container and GPU instance are removed from the deployment.

Implementation:
```python
client = genai.Client(api_key=GEMINI_API_KEY)
config = GenerateContentConfig(
    system_instruction=TAMIL_ANSWER_SYSTEM_PROMPT,
    temperature=0.0, max_output_tokens=4096, top_p=0.9,
    thinking_config=ThinkingConfig(thinking_budget=0),
)
```

**Consequences:** Response time dropped from 15–20s to 2–5s. Infrastructure cost dropped from $120–386/month to effectively $0 (Gemini free tier is sufficient for current single-user scale; pay-per-use is ~$0.15/1M input tokens). Tamil generation quality improved significantly — Gemini correctly handles metadata accuracy, multi-record queries, article type classification, and title disambiguation where Tamil LLaMA failed or hallucinated. Native streaming support via `generate_content_stream()` enables SSE. The deployment simplified — no GPU instance, no NVIDIA Docker toolkit, no model lifecycle management. The Ollama container was removed from `docker-compose.yml` (4 services → 3).

The trade-off is vendor dependency on Google's API, a lack of offline capability, and a medium hallucination risk (Gemini occasionally provides detailed but fabricated context). If Gemini's free tier becomes insufficient, pay-per-use costs scale linearly with usage. The self-hosted option remains viable as a fallback (the Modelfile is retained in the repository).

Lessons learned:
1. Always verify GPU layer offloading (`num_gpu`) — the default of 1 is a silent performance killer
2. Ollama's default 10-minute `keep_alive` is too aggressive for interactive apps
3. 7B parameter models are insufficient for literary Tamil document analysis — initial evaluation on 10 generic questions masked production-level quality gaps in metadata accuracy and multi-record handling
4. Local LLM infrastructure maintenance cost (time + money) exceeds API costs at small scale
5. Head-to-head comparative testing on production-representative queries (not just generic QA) is essential before selecting a model

---

### ADR-005: Dual-Prompt System

**Status:** Accepted

**Context:** The system handles two fundamentally different query types: structured data queries (authors, topics, issues from a CSV metadata file) and unstructured document queries (literary content from vector search). A single universal prompt would bloat context for simple CSV queries and provide insufficient instruction depth for complex document queries.

**Decision:** We will use two separate LLM prompt pipelines routed by query type detection.

**CSV Queries** use `_CSV_SYSTEM_PROMPT` + `_build_csv_user_content()`:
- Target: 30–150 words
- Question-type-aware closing (yes/no → "ஆம்/இல்லை first", wh-question → "direct answer first", list → "summarize trends")
- Output: LLM gist + raw CSV data appended below separator
- Thinking disabled (`thinking_budget=0`) for speed

**Vector Search Queries** use `TAMIL_ANSWER_SYSTEM_PROMPT` + `_build_user_content()`:
- Target: 200–500 words
- Includes full Ponni magazine background context (founding, timeline, contributors, significance)
- Question repeated after context block (improves model attention, inspired by arxiv 2512.14982)
- Multi-document integration instruction when N > 1 docs
- Fallback: extractive answer from top-5 keyword-matching sentences if LLM fails

**Consequences:** Answer quality improved for both query types — CSV queries are concise and direct, document queries are comprehensive with background context. The trade-off is more complex routing logic and two sets of prompts to maintain. Adding a new query type requires adding a new prompt pipeline. No additional cost — same Gemini API, different prompts.

---

### ADR-006: Frontend — Streamlit to React + FastAPI

**Status:** Accepted (supersedes Streamlit as primary UI)

**Context:** The initial Streamlit UI was fast to prototype but hit limitations as the application matured. Streamlit's server-side rendering model reruns the entire script on every interaction, making SSE streaming impossible and limiting interactivity. The library/PDF viewer and bilingual UI required more control than Streamlit widgets provide. Alternatives considered: keeping Streamlit (too limited), Next.js + FastAPI (heavier framework, SSR not needed).

**Decision:** We will build a React 18 SPA with a FastAPI REST backend, served via an Nginx reverse proxy. Streamlit is retained for local development testing.

React stack: React 18.2, react-router-dom 6.20, axios 1.6, react-markdown 10.1, bilingual UI via `translations.js`. Pages: Home (Search), Library (Volumes → Issues → PDF), Tag Browse (Phase 1.1), About.

FastAPI stack: Uvicorn ASGI server (2 workers), Pydantic validation. Endpoints: `/api/ask` (+ tag filter), `/api/ask/stream` (SSE), `/api/search` (+ tags), `/api/tags`, `/api/tags/{id}/articles` (Phase 1.1), `/api/authors`, `/api/topics`, `/api/library/*`, `/api/cache/stats`, `/health`.

**Consequences:** SSE streaming works natively — users see tokens arrive in real-time instead of waiting for the full response. The frontend and backend can be developed and scaled independently. Modern navigation (client-side routing) replaces Streamlit's page reload model. The trade-off is more development effort (separate React build pipeline, Nginx configuration) and two codebases to maintain. No additional deployment cost — Nginx serves static assets from the same Docker Compose stack.

---

### ADR-007: Streaming Architecture

**Status:** Accepted

**Context:** Gemini responses take 2–5s to generate fully. Without streaming, users see no feedback during this time — the UI appears frozen. Alternatives considered: WebSockets (bidirectional, but overkill for unidirectional LLM output), polling (simplest but high latency and wasteful requests).

**Decision:** We will use Server-Sent Events (SSE) via FastAPI's `StreamingResponse`.

The `/api/ask/stream` endpoint wraps Gemini's `generate_content_stream()` — each token is sent as an SSE `data:` event. Nginx is configured with `proxy_buffering off`, `proxy_read_timeout 300s`, and `chunked_transfer_encoding on`. The frontend uses the browser's `EventSource` API. A non-streaming fallback (`/api/ask`) returns complete JSON for environments without SSE support.

**Consequences:** Perceived latency drops significantly — users see the first token within ~500ms instead of waiting 2–5s for the full response. SSE is simple to implement (standard HTTP, no connection upgrade) and has native browser support. The trade-off is unidirectional communication only (server → client), which is exactly the pattern needed for LLM streaming. The Nginx proxy configuration must explicitly disable buffering, or SSE events are batched and delivered all at once.

---

### ADR-008: Response Caching

**Status:** Accepted

**Context:** Identical or near-identical questions should not hit the Gemini API every time. The system needs a caching layer to reduce latency and API usage. Alternatives considered: Redis (persistent, shared across workers — but overkill for a single-user, single-process app), disk-based caching (I/O overhead, concurrency issues).

**Decision:** We will use an in-memory TTL + LRU cache implemented as a `ResponseCache` class.

```python
class ResponseCache:
    max_size = 100        # LRU entries
    ttl_seconds = 3600    # 1-hour TTL
    key = SHA256(normalize(question))  # case-insensitive, whitespace-collapsed
```

Thread-safe via `threading.Lock`. `OrderedDict` for O(1) LRU operations. Hit/miss statistics exposed via `/api/cache/stats`.

**Consequences:** Cached responses return in <100ms vs 2–5s uncached. The SHA256 key normalization (lowercase, whitespace-collapsed) means minor phrasing differences in the same question still hit the cache. The trade-off is that the cache is lost on restart — acceptable because the app restarts rarely and the cache warms quickly with usage. The cache is single-process only; if Uvicorn runs multiple workers, each has its own cache. At current scale (single user, 2 workers) this is not a problem.

---

### ADR-009: Document Processing Pipeline

**Status:** Accepted

**Context:** Raw documents (PDFs, DOCX) stored in S3 need to be chunked, embedded, and indexed into Qdrant for retrieval. The chunking strategy must balance search granularity (smaller chunks match more precisely) against context coherence (larger chunks provide more readable evidence for the LLM).

**Decision:** We will use 500-character sentence-aware chunks at indexing time, with consecutive chunk merging at query time.

Indexing pipeline:
```
S3 (ponni-dev/) → text_extraction.py → article_seperation.py → content_extraction.py
    → qdrant_indexer.py (2-pass: tagger training + tag/chunk/embed) → Qdrant (qdrant_indexer)
```

Key parameters: 500-char chunks, batch upsert of 100, E5 `passage:` prefix for indexing / `query:` for search, 1024-dim dense + BM25 sparse vectors in one collection. Phase 1.1 added article tagging during indexing — Pass 1 collects articles for tagger training, Pass 2 tags each article (1–3 categories) then chunks, embeds, and upserts with `tags` and `tags_tamil` in the Qdrant payload.

At query time: consecutive chunks from the same document are merged (`merge_consecutive_chunks`), documents with < 50 words are filtered out, a 15,000-character context budget is distributed equally across relevant docs (min 500 chars each, max 15 docs), and keyword-relevant excerpts are extracted via sliding window (not just first N chars).

**Consequences:** Small chunks provide precise retrieval — a 500-char chunk about a specific author scores higher than a full 5,000-char article. The merging step reconstructs full documents for coherent LLM context and user-facing evidence. The trade-off is additional latency at query time (each unique document requires a scroll query to retrieve all its chunks). The keyword-relevant excerpt extraction means the LLM sees the most question-relevant portion of each document, not just the beginning. Storage cost is minimal (one-time indexing).

---

### ADR-010: Deployment — Docker Compose

**Status:** Accepted

**Context:** The system needs a reproducible, portable deployment for the full stack (vector database, API backend, frontend). The deployment originally included 4 services (Qdrant, Ollama, FastAPI, React/Nginx) but Ollama was removed after the Gemini migration.

**Decision:** We will use Docker Compose with 3 services.

```yaml
services:
  qdrant:      # Port 6333 — vector database, TCP healthcheck, persistent volume
  api:         # Port 8000 — FastAPI, depends_on qdrant (service_healthy)
  frontend:    # Port 3000→80 — React + Nginx reverse proxy, depends_on api
```

Nginx serves as reverse proxy: `/` → React SPA (try_files → index.html), `/api/*` → proxy_pass to FastAPI, `/health` → health check proxy. SSE support: `proxy_buffering off`, `proxy_read_timeout 300s`. Static assets cached with 1-year `Cache-Control: public, immutable`. Gzip compression for text/CSS/JS/JSON.

**Consequences:** The entire stack starts with a single `docker-compose up --build -d` command. Service dependencies are enforced (API waits for Qdrant health check before starting). The Ollama removal simplified the deployment — no GPU runtime, no NVIDIA Docker toolkit, no model volume mounts, no g4dn.xlarge instance. The trade-off is that Docker Compose is single-host only; multi-host scaling would require migrating to Docker Swarm or Kubernetes. This is acceptable for the current single-user research tool. Docker itself is free; EC2 hosting costs depend on instance type (see Cost Summary).

---

### ADR-011: Tamil NLP Decisions

**Status:** Accepted

**Context:** Tamil has unique linguistic properties that break common English-centric NLP assumptions. `\b` (regex word boundary) does not work with Tamil Unicode characters (U+0B80–U+0BFF). Tamil names have honorific prefixes (கவியரசு, பாவேந்தர், கலைஞர்), possessive suffixes (னின், ரின், யின்), and agglutinative morphology. Author names in the CSV metadata appear in many variant forms. Standard NER tools do not handle these cases reliably.

**Decision:** We will build custom Tamil NLP utilities in `tamil_text.py`, including:

1. **Tamil-safe regex:** All word boundary patterns use `(?=\s|$)` instead of `\b`.
2. **Six-tier author fuzzy matching:** exact → substring → alias → token-level → fuzzy 95% → edit distance ≤ 1.
3. **Tamil suffix stripping:** Order-sensitive rules (longer suffixes first) for possessive/genitive/locative forms.
4. **Canonical author mapping:** Maps honorifics, typos, and name variants to canonical forms.

```python
# WRONG: \b does not work with Tamil
re.sub(r'னின்\b', 'ன்', text)

# CORRECT: explicit lookahead
re.sub(r'னின்(?=\s|$)', 'ன்', text)
```

**Consequences:** Author matching works correctly across name variants — "கலைஞர்", "கருணாநிதி", and "மு. கருணாநிதி" all resolve to the same author. Tamil suffix stripping enables accurate topic and entity extraction from inflected query text. The critical bug where honorific-only names (e.g., 'பாவேந்தர்') were stripped to empty string — causing every CSV row to match — was caught and fixed.

The trade-off is that hand-crafted rules require maintenance as new edge cases are discovered. Adding a new author alias or suffix rule is a manual process. However, this approach is more predictable and debuggable than ML-based NER for this specific closed domain of ~200 known authors.

---

### ADR-012: Evaluation Framework

**Status:** Accepted

**Context:** Standard English NLP evaluation metrics (word-level BLEU, ROUGE) underperform on Tamil because Tamil is agglutinative — a single written "word" may encode subject, object, tense, and aspect. Morphological variants of the same root look completely different at the word level but share most of their characters. The system needs Tamil-appropriate metrics to measure answer quality.

**Decision:** We will use a composite metric with character-level tokenization, weighted as: semantic similarity (50%) + BERTScore-F1 (25%) + ROUGE-L (15%) + BLEU-1 (10%).

All lexical metrics (BLEU, ROUGE-L) operate at the character level — each Unicode code-point is treated as a token. BERTScore uses `intfloat/multilingual-e5-large` (the same model already deployed for search) to avoid a second large model download. Unicode NFC normalization ensures consistent code-point sequences. Punctuation is preserved because Tamil punctuation carries sentence-boundary information.

Implementation: `src/evaluation/metrics.py` — `MetricsCalculator` class with lazy model loading, batch embedding support, and per-sample `MetricResult` dataclass.

**Consequences:** Semantic similarity dominates (50%) because Tamil synonyms and inflections are common — exact-match lexical scores alone would underestimate answer quality. Character-level BLEU/ROUGE captures morphological overlap that word-level metrics miss. Reusing the E5-large model for BERTScore eliminates an extra ~2.3 GB model download.

The trade-off is that character-level metrics can over-credit superficially similar but semantically different text. The 50% semantic similarity weight mitigates this. BERTScore adds token-level precision at the cost of additional compute (one model forward pass per evaluation pair).

---

### ADR-013: Cross-Encoder Reranking (Phase 1.1)

**Status:** Accepted

**Context:** The Phase 1 score-gap filtering (absolute floor at 35% of top score, gap detection at 40% of previous) was a heuristic that worked reasonably but had failure modes: (1) semantically irrelevant documents with high embedding similarity passed through, (2) relevant documents with slightly lower embedding scores were cut too aggressively, and (3) the threshold values required manual tuning per query type. A cross-encoder reranker that scores query-document pairs jointly can make more informed relevance decisions.

**Decision:** We will add a cross-encoder reranking stage after RRF fusion, using `BAAI/bge-reranker-v2-m3` (multilingual, supports Tamil).

The reranking pipeline:
```
RRF fused results (top 30) → Heading-phrase pinning → Cross-encoder scoring → Adaptive cutoff
```

Key features:
- **Heading-phrase pinning:** If the query phrase exactly matches an article title (≥2 words), that article is pinned to the top and skips the cross-encoder — avoids misjudging exact matches. Max 3 pinned docs.
- **Clear-win skip:** When a small number of documents dominate the heading match, the cross-encoder is skipped entirely — faster response with no quality loss.
- **Adaptive score cutoff:** Floor at 40% of top CE score, gap cutoff at 50% of previous score. Min 3, max 5 results.
- **All thresholds env-configurable:** `RERANKER_TOP_IN`, `RERANKER_TOP_OUT`, `RERANKER_SCORE_FLOOR_RATIO`, `RERANKER_GAP_RATIO`, etc.

```python
RERANKER_MODEL     = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_IN    = 30       # candidates into reranker
RERANKER_TOP_OUT   = 5        # results out of reranker
RERANKER_MAX_CHARS = 800      # per-doc truncation
RERANKER_BATCH     = 16       # batch size for CE scoring
```

**Consequences:** Precision improved significantly — the cross-encoder understands semantic relationships between query and document content, not just vector distance. Heading-phrase pinning handles the common case where users search by article title directly (zero CE latency for those). The `ENABLE_RERANKER=0` env var provides a kill switch to fall back to the previous score-gap approach if needed. The trade-off is ~200ms additional latency per query (30 candidates × 800 chars each, batched at 16), plus a ~1.1 GB model download. The reranker runs on CPU — no GPU required. No API cost.

---

### ADR-014: BM25 Sparse Embeddings via fastembed (Phase 1.1)

**Status:** Accepted (supersedes MD5 hash approach)

**Context:** Phase 1 used deterministic MD5 token hashing for sparse vectors — a custom implementation that mapped each unique token to a fixed hash bucket. This worked but had limitations: (1) no IDF weighting, so common Tamil stopwords had the same influence as rare literary terms, (2) hash collisions in a fixed-size space degraded precision, and (3) the approach was non-standard and harder to maintain.

**Decision:** We will replace the MD5 hash approach with `Qdrant/bm25` via the fastembed library, which provides proper BM25 sparse embeddings with IDF weighting.

Implementation uses separate doc/query embedding functions:
- `sparse_embed_doc(text)` — document-side weights (TF × IDF)
- `sparse_embed_query(text)` — query-side weights (unit TF, IDF only)

Controlled by the same `ENABLE_RERANKER` flag (both features were introduced together).

**Consequences:** Sparse search quality improved — IDF weighting means rare Tamil literary terms and proper nouns get higher weight than common words, which is exactly the behavior needed for BM25-style retrieval. The fastembed library is maintained by the Qdrant team, ensuring compatibility with the Qdrant sparse vector format. The trade-off is an additional ~50 MB model download and slightly higher indexing time. No API cost.

---

### ADR-015: Article Tagging System (Phase 1.1)

**Status:** Accepted

**Context:** The 1,200+ articles in the Ponni archive had no categorical classification. Users could search by keyword or author but could not browse by topic (fiction, poetry, politics, etc.). An LLM-based tagging approach was considered but rejected: (1) API cost for 1,200+ articles, (2) non-deterministic results, (3) unnecessary for a closed taxonomy. The content is in Tamil literary prose from a specific era (1947–1955), so general-purpose NER/classification tools perform poorly.

**Decision:** We will build a hybrid rule-based + TF-IDF tagger (`article_tagger.py`) with a fixed 15-category taxonomy, fully offline and deterministic.

```
TAXONOMY (15 categories):
  FICTION, EDITORIAL, LITERARY_REVIEW, POETRY, QA_COLUMN,
  ARTS_CULTURE, SATIRE_HUMOR, CLASSICAL_LIT, PUBLIC_FORUM,
  WOMENS_ISSUES, RELIGIOUS_DEBATE, CHILDRENS, POLITICAL,
  BIOGRAPHY, GENERAL
```

The tagging pipeline:
1. **Rule-based:** Keyword patterns per category matched against title + body. High-confidence, handles known patterns (serial fiction titles, editorial headers, poetry markers).
2. **TF-IDF fallback:** For articles not caught by rules, trained on the labeled corpus via cosine similarity. Threshold: 0.15 minimum.
3. **Serial detection:** Multi-part fiction detection across issues (same title, different issue numbers).
4. Each article receives 1–3 tags.

Tags are stored in Qdrant payload (`metadata.tags`, `metadata.tags_tamil`) and synced to `summary.csv` (`வகை` column) via `update_csv_tags.py`.

**Consequences:** All articles are classified without API cost. The deterministic approach means re-indexing produces identical tags — important for reproducibility. The Tag Browse page (`/tags`) uses these tags for filtering. The `/api/ask` and `/api/search` endpoints accept optional `tags` parameter to restrict search to specific categories. The trade-off is that the 15-category taxonomy is fixed — adding a new category requires updating `TAXONOMY`, adding keyword rules, and re-indexing. The TF-IDF fallback covers edge cases but may misclassify articles with unusual vocabulary (threshold 0.15 is conservative to minimize false positives).

---

### ADR-016: Qdrant Snapshot Management (Phase 1.1)

**Status:** Accepted

**Context:** Qdrant data lives in a Docker volume. If the volume is lost (container rebuild, host migration), the entire index must be regenerated — a process that takes 30+ minutes (S3 download + text extraction + embedding + indexing). The system needed a way to persist and restore the index state.

**Decision:** We will use Qdrant's native snapshot API with S3 backup/restore, implemented in `snapshot_manager.py`.

Key features:
- **Embedding fingerprint:** SHA-256 hash of `model_name + embedding_dim + chunk_size`. If the fingerprint changes (e.g., model upgrade), a full re-index is triggered instead of restoring a stale snapshot.
- **Source data hash:** SHA-256 hash of S3 file keys + last-modified timestamps. Detects when source documents have been added/modified.
- **Automatic restore:** On startup, if the Qdrant collection is empty but a valid snapshot exists in S3, it is restored automatically.

**Consequences:** Recovery from volume loss takes ~2 minutes (S3 download + Qdrant restore) instead of 30+ minutes (full re-index). The fingerprint check prevents restoring a snapshot built with a different embedding model — avoiding silent search quality degradation. The trade-off is S3 storage cost for snapshots (~50 MB compressed) and the need to upload a new snapshot after each re-index.

---

### ADR-017: Tag Browse Frontend Page (Phase 1.1)

**Status:** Accepted

**Context:** The Tag Browse page (`/tags`) was needed to allow users to explore the archive by category rather than only by keyword search. The page needed to work with the existing volume/issue/article hierarchy while adding a category dimension.

**Decision:** We will build a `TagBrowse.js` page with a left sidebar (volume accordion + category filter + search) and right content pane (article list → article detail).

API calls:
- `GET /api/tags` — loads full taxonomy with article counts
- `GET /api/tags/{id}/articles` — articles bearing a specific tag
- `GET /api/library/volumes/{id}/issues/{id}/articles` — articles per issue
- `GET /api/library/articles/{docId}/{docIssue}/{articleNo}` — full article content

Key design choices:
- Volume accordion with lazy-loading (issues fetched on expand, not on page load)
- Client-side filtering by category and search text (no additional API calls after initial load)
- Three-state content pane: empty → article list → article detail

**Consequences:** Users can browse the full archive by category, volume, or a combination. The lazy-loading approach keeps initial page load fast. The client-side filtering is responsive for the current corpus size (~1,200 articles). The trade-off is that the category filter is fetched from `/api/tags` which requires a Qdrant scroll — if the index is unavailable, the page falls back to the full taxonomy with zero counts.

---

## Cost Summary

### Compute Costs

| Component | Self-Hosted (Ollama era) | Current (Gemini era) |
|-----------|--------------------------|----------------------|
| **LLM Inference** | | |
| — g4dn.xlarge (On-Demand) | $0.526/hr → ~$384/mo | Not needed |
| — g4dn.xlarge (Spot) | ~$0.16–0.20/hr → ~$120–146/mo | Not needed |
| — g4dn.xlarge (1yr Reserved, All Upfront) | ~$0.27/hr → ~$197/mo | Not needed |
| — Gemini 2.5 Flash API | — | Free tier (sufficient at current scale) |
| — Gemini pay-per-use (if exceeding free tier) | — | ~$0.15/1M input tokens, ~$0.60/1M output tokens |
| **Vector DB (Qdrant)** | | |
| — Self-hosted (Docker) | Free | Free |
| — EBS gp3 persistence (30 GB) | ~$2.40/mo | ~$2.40/mo |
| **S3 Storage** | | |
| — ponni-dev bucket (PDFs, DOCX, JSON) | ~$0.023/GB/mo (minimal) | ~$0.023/GB/mo (minimal) |
| **EC2 for API + Frontend** | | |
| — t3.medium (no GPU needed now) | — | ~$0.0416/hr → ~$30/mo |
| — t3.medium (Spot) | — | ~$0.0125/hr → ~$9/mo |

### Monthly Cost Comparison

| Scenario | Ollama Self-Hosted | Gemini API (Current) |
|----------|-------------------|----------------------|
| Minimum (spot, 12hr/day) | ~$72 + $2.40 = **~$74** | ~$9 + $2.40 = **~$11** |
| Standard (spot, 24/7) | ~$120–146 + $2.40 = **~$122–148** | ~$9 + $2.40 = **~$11** |
| On-Demand (24/7) | ~$384 + $2.40 = **~$386** | ~$30 + $2.40 = **~$32** |

---

## What Worked / What Didn't

| Worked | Didn't Work |
|--------|-------------|
| Hybrid search (dense + sparse + RRF) — excellent Tamil retrieval quality | Ollama/Tamil-Llama 7B local LLM — too slow (15–20s even with GPU fix) |
| `intfloat/multilingual-e5-large` embeddings — strong Tamil support | `num_gpu: 1` default in Modelfile — only 1/32 layers on GPU, CPU-bound |
| Qdrant self-hosted — native hybrid search, zero API cost | 10-minute model eviction — cold start reload on every idle period |
| Gemini 2.5 Flash API — 2–5s responses, no infrastructure | Streamlit as production UI — server-side rendering, no streaming |
| React + FastAPI decoupled architecture — SSE streaming, modern UX | Monolithic `hybrid_search.py` — grew to unmanageable size before split |
| Docker Compose (3 services) — reproducible, portable | `\b` word boundary in Tamil regex — ASCII-only, breaks on Unicode |
| Response caching (TTL+LRU) — <100ms for repeated queries | MD5 hash for sparse vectors (Phase 1) — no IDF weighting, hash collisions |
| Dual-prompt system — tailored instructions per query type | Honorific names in alias matching — empty string matches everything |
| Character-level BLEU/ROUGE for Tamil evaluation | Score-gap relevance filtering (Phase 1) — heuristic, replaced by reranker |
| Code modularization (5 sub-modules) — maintainable, testable | |
| Cross-encoder reranking (Phase 1.1) — much better precision than score-gap | |
| BM25 via fastembed (Phase 1.1) — proper IDF weighting, standard approach | |
| Article tagging (Phase 1.1) — deterministic, no LLM cost, 15 categories | |
| Qdrant snapshots to S3 (Phase 1.1) — 2-min recovery vs 30-min re-index | |
| Heading-phrase pinning (Phase 1.1) — zero-latency for exact title matches | |

---

## Current Architecture Summary

### Service Map

```
┌──────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                    │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │   Frontend    │   │   FastAPI    │   │     Qdrant       │ │
│  │  (Nginx)     │──►│   (API)      │──►│  (Vector DB)     │ │
│  │  Port 3000   │   │  Port 8000   │   │  Port 6333       │ │
│  └──────────────┘   └──────┬───────┘   └──────────────────┘ │
│                             │                                 │
│                             ▼                                 │
│                    ┌──────────────────┐                       │
│                    │ Gemini 2.5 Flash │ (External API)        │
│                    │ (Google Cloud)   │                       │
│                    └──────────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

### Port Map

| Port | Service | Access |
|------|---------|--------|
| 3000 | Nginx (React frontend) | Public — user-facing |
| 8000 | FastAPI (API) | Internal — proxied via Nginx `/api/*` |
| 6333 | Qdrant | Internal — API container only |

### Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | api | Google Gemini API key |
| `GEMINI_MODEL` | api | LLM model (default: `gemini-2.5-flash`) |
| `QDRANT_HOST` | api | Qdrant hostname (default: `qdrant`) |
| `QDRANT_PORT` | api | Qdrant port (default: `6333`) |
| `AWS_ACCESS_KEY_ID` | api | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | api | S3 secret key |
| `ENABLE_RERANKER` | api | Enable cross-encoder reranking (default: `1`) |
| `RERANKER_MODEL` | api | Reranker model (default: `BAAI/bge-reranker-v2-m3`) |
| `RERANKER_TOP_IN` | api | Candidates fed to reranker (default: `30`) |
| `RERANKER_TOP_OUT` | api | Results kept after reranking (default: `5`) |
| `RAG_DEBUG` | api | Debug logging flag (default: `0`) |

### Key Constants

```python
COLLECTION_NAME    = "qdrant_indexer"
EMBEDDING_MODEL    = "intfloat/multilingual-e5-large"
EMBEDDING_DIM      = 1024
CHUNK_SIZE         = 500        # chars per chunk
BATCH_SIZE         = 100        # upsert batch
SCORE_THRESHOLD    = 0.65       # min cosine for dense prefetch
MAX_QUERY_LENGTH   = 500        # query truncation limit
CONTEXT_BUDGET     = 15000      # chars distributed across docs
CACHE_MAX_SIZE     = 100        # LRU entries
CACHE_TTL          = 3600       # 1 hour
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_TOKENS  = 4096
GEMINI_TOP_P       = 0.9

# Phase 1.1 additions
RERANKER_MODEL            = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_IN           = 30
RERANKER_TOP_OUT          = 5
RERANKER_SCORE_FLOOR_RATIO = 0.4
RERANKER_GAP_RATIO        = 0.5
TFIDF_THRESHOLD           = 0.15    # article tagging
TAXONOMY_SIZE             = 15      # article categories
```
