"""
Hybrid Search Module for Tamil Document Processing.
Provides vector and keyword-based search with Gemini LLM-powered answer generation.
Embedding model loads on GPU if available, falls back to optimized CPU.
"""
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict, OrderedDict
import re
import os
import logging
from pathlib import Path
import streamlit as st

import torch
torch.set_grad_enabled(False)
from qdrant_client import models
import pandas as pd
import requests
import json
import asyncio
import threading
import httpx
import time
import hashlib

from google import genai
from google.genai import types as genai_types

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    # CPU optimizations: use all but one core for torch, 2 for interop
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # Already configured by another module

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "qdrant_indexer"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "summary.csv"
SCORE_THRESHOLD = 0.8  # Minimum cosine similarity for dense vector search

_embed_lock = threading.Lock()


class ResponseCache:
    """Thread-safe TTL + LRU cache for ask_question results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(question: str) -> str:
        normalized = re.sub(r'\s+', ' ', question.strip().lower())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
        key = self._make_key(question)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            if time.time() - self._timestamps.get(key, 0) > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                logger.info(f"[CACHE] TTL expired for {key[:12]}...")
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            total = self._hits + self._misses
            logger.info(f"[CACHE] HIT (hits={self._hits}, misses={self._misses}, rate={self._hits / total:.0%})")
            return self._cache[key]

    def put(self, question: str, result: Dict) -> None:
        key = self._make_key(question)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = result
                self._timestamps[key] = time.time()
                return
            while len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(evicted_key, None)
            self._cache[key] = result
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.info("[CACHE] Cache cleared")

    def stats(self) -> Dict:
        with self._lock:
            total = max(1, self._hits + self._misses)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total:.0%}",
            }


_response_cache = ResponseCache(max_size=100, ttl_seconds=3600)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_files():
    """Verify existence of critical files (CSV)."""
    logger.info("Verifying critical files...")
    if CSV_PATH.exists():
        logger.info(f"CSV found: {CSV_PATH}")
        logger.info(f"CSV size: {CSV_PATH.stat().st_size / 1024:.2f} KB")
    else:
        logger.error(f"CSV NOT FOUND: {CSV_PATH}")


verify_files()


# ============================================================================
# STREAMLIT-CACHED MODEL LOADERS (PERSISTENT ACROSS RERUNS)
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_embed_model():
    """
    Load and cache embedding model using Streamlit's cache_resource.
    This ensures the model loads ONCE and persists across all reruns.
    Spinner is disabled - will only show during preload_models().
    """
    logger.info(f"Loading embedding model on {DEVICE} (this happens only once)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    logger.info(f"Embedding model loaded on {DEVICE} and cached")
    return model


@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    """
    Get or initialize Qdrant client (singleton pattern with Streamlit caching).
    Spinner is disabled - will only show during preload_models().
    """
    logger.info("🔄 Connecting to Qdrant SERVER...")

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=False,
        timeout=30.0
    )

    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Connected: {collection_info.points_count} points")

    return client


@st.cache_resource(show_spinner=False)
def get_csv_dataframe():
    """Load CSV once and cache it."""
    if not CSV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource(show_spinner=False)
def get_csv_embeddings():
    """
    Precompute embeddings for CSV rows.
    Cached permanently like embedding model.
    """
    df = get_csv_dataframe()
    model = get_embed_model()

    if df.empty:
        return []

    texts = []
    for _, row in df.iterrows():
        text = " | ".join([str(v) for v in row.values if pd.notna(v)])
        texts.append(text)

    embeddings = model.encode(
        [f"passage: {t}" for t in texts],
        show_progress_bar=False
    )

    return list(zip(texts, embeddings))


# ============================================================================
# EMBEDDING FUNCTIONS (USE CACHED MODELS)
# ============================================================================

def dense_embed_query(text: str):
    """Generate dense embedding for query text (thread-safe)."""
    model = get_embed_model()
    with _embed_lock:
        return model.encode(f"query: {text}").tolist()


def _deterministic_token_hash(token: str) -> int:
    """Deterministic token hash using MD5, consistent across processes.

    Python's built-in hash() is randomized per process (PYTHONHASHSEED),
    which causes sparse vectors at query time to mismatch those created
    at indexing time.  MD5 is deterministic and fast for this use case.
    """
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % (2**31)


def sparse_embed(text: str):
    """Generate sparse BM25-style embedding for text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    indices, values = [], []
    for token, freq in counts.items():
        indices.append(_deterministic_token_hash(token))
        values.append(float(freq))
    return models.SparseVector(indices=indices, values=values)


def search_csv_semantic(question: str, top_k: int = 5):
    """Semantic search over CSV rows."""
    csv_data = get_csv_embeddings()
    model = get_embed_model()

    if not csv_data:
        return []

    query_emb = model.encode(f"query: {question}")

    scored = []
    for text, emb in csv_data:
        score = float(torch.tensor(query_emb) @ torch.tensor(emb))
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [text for text, _ in scored[:top_k]]


# ============================================================================
# QDRANT HEALTH & HYBRID SEARCH
# ============================================================================

def check_qdrant_health() -> Dict:
    """Check Qdrant database health and connectivity."""
    try:
        client = get_qdrant_client()

        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            return {
                "healthy": True,
                "collection": COLLECTION_NAME,
                "points_count": collection_info.points_count,
                "message": "Qdrant server is healthy"
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": "collection_not_found",
                "message": f"Collection '{COLLECTION_NAME}' not found",
                "details": str(e),
                "action": "Create the collection on the server"
            }
    except Exception as e:
        return {
            "healthy": False,
            "error": "connection_failed",
            "message": "Failed to connect to Qdrant server",
            "details": str(e),
            "action": "Ensure Qdrant server is running on port 6333"
        }


class HybridQdrantSearch:
    """Hybrid search combining dense and sparse vectors for optimal results."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def search(self, query: str, limit: int = 30, score_threshold: float = SCORE_THRESHOLD):
        """
        Perform hybrid search using dense and sparse vectors.

        Executes a two-stage search combining dense embeddings for semantic similarity
        and sparse embeddings for keyword matching, then fuses results using RRF.

        Dense prefetch uses a cosine score_threshold (default 0.8) to gate
        semantic quality.  Sparse prefetch is capped at `limit` (not limit*2)
        to avoid flooding the RRF pool with weak keyword matches.

        Args:
            query (str): Search query string in Tamil or English
            limit (int, optional): Maximum number of results to return. Defaults to 30.
            score_threshold (float, optional): Minimum cosine similarity for dense
                vector results. Defaults to SCORE_THRESHOLD (0.8).

        Returns:
            List[ScoredPoint]: List of scored points from Qdrant with fused relevance scores
        """
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_embed_query(query),
                    using="dense",
                    filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="article"))]),
                    score_threshold=score_threshold,
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=sparse_embed(query),
                    using="sparse",
                    filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="article"))]),
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points


# ============================================================================
# ORCHESTRATOR FUNCTIONS
# ============================================================================

def ask_question(question: str, return_formatted: bool = False, use_llm: bool = True) -> Dict:
    """
    Main question answering function with database health check and hybrid search.

    Primary entry point for processing user queries. Handles database health checks,
    author queries, vector search, document merging, LLM generation, and source formatting.
    Results are filtered by score threshold rather than a fixed top_k count.
    CSV queries are checked BEFORE vector search to return direct data without LLM.

    Args:
        question (str): User's question in Tamil or English.
        return_formatted (bool, optional): If True, return formatted string; if False,
                                        return dict. Defaults to False.
        use_llm (bool, optional): If True, use LLM for answer generation; if False,
                                use extractive fallback only. Defaults to True.

    Returns:
        dict or str: Depending on return_formatted:
            - If False (default): Dict with keys:
                - answer (str): Generated answer text
                - sources (list): List of source document dicts
                - query_type (str): Type of query handled (optional)
                - error (str/dict): Error details if failed (optional)
            - If True: Formatted string with answer and sources
    """
    t_start = time.time()
    health_status = check_qdrant_health()
    logger.info(f"[TIMING] health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        error_message = f"""
Database Error: {health_status['message']}

Details: {health_status.get('details', 'No additional details')}
Action: {health_status.get('action', 'Contact administrator')}

Error Type: {health_status['error']}
"""
        if return_formatted:
            return error_message
        return {
            "answer": error_message,
            "sources": [],
            "error": health_status
        }

    logger.info("Database is healthy - proceeding with query")

    # --- RESPONSE CACHE LOOKUP ---
    cached = _response_cache.get(question)
    if cached is not None:
        logger.info(f"[TIMING] TOTAL ask_question (CACHED): {time.time() - t_start:.2f}s")
        if return_formatted:
            return format_answer_output(cached["answer"], cached["sources"])
        return cached
    # --- END CACHE LOOKUP ---

    # 2. CHECK CSV QUERIES FIRST (BEFORE VECTOR SEARCH)
    if CSV_PATH.exists():
        try:
            logger.info("Checking author query...")
            is_handled, csv_response = handle_author_query(question, str(CSV_PATH))

            if is_handled and csv_response and csv_response.strip():
                logger.info("CSV query matched - passing to LLM for summarization")
                t0 = time.time()
                csv_content = _build_csv_user_content(question, csv_response)
                llm_summary = generate_llm_answer(
                    question, context="", csv_context="",
                    user_content=csv_content, system_prompt=_CSV_SYSTEM_PROMPT,
                    disable_thinking=True,
                )
                logger.info(f"[TIMING] gemini_llm (csv): {time.time() - t0:.2f}s")
                logger.info(f"CSV LLM gist length: {len(llm_summary or '')} chars")

                if not llm_summary or len(llm_summary) < 5:
                    llm_summary = ""

                # Combine: LLM gist + raw CSV data appended
                combined_answer = _combine_csv_answer(llm_summary, csv_response)

                result = {
                    "answer": combined_answer,
                    "sources": [],
                    "query_type": "author_csv",
                }
                _response_cache.put(question, result)

                if return_formatted:
                    return format_answer_output(combined_answer, [])
                return result
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    # 3. IF NOT CSV QUERY, PROCEED WITH VECTOR SEARCH + LLM
    try:
        client = get_qdrant_client()

        logger.info(f"Searching: {question[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=200, score_threshold=SCORE_THRESHOLD)
        logger.info(f"[TIMING] hybrid_search: {time.time() - t0:.2f}s ({len(results)} results)")

        if not results:
            answer = "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        t0 = time.time()
        merged_docs = merge_consecutive_chunks(client, results)
        logger.info(f"[TIMING] merge_chunks: {time.time() - t0:.2f}s ({len(merged_docs)} docs)")

        if not merged_docs:
            answer = "போதுமான தகவல்கள் இல்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        logger.info(f"Merged into {len(merged_docs)} documents")

        # Select relevant docs (same set used for context + evidence)
        relevant_docs = _select_relevant_docs(merged_docs)
        logger.info(f"Selected {len(relevant_docs)} relevant documents for context")

        # Build context with equal excerpts from all relevant docs
        context, context_doc_count = build_context_from_docs(relevant_docs, question)

        # CSV semantic context
        csv_results = search_csv_semantic(question, top_k=3)
        logger.info("------ CSV Rows Sent To LLM ------")
        for row in csv_results:
            logger.info(row)
        logger.info("----------------------------------")

        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])

        answer = ""
        if use_llm:
            t0 = time.time()
            answer = generate_llm_answer(
                question, context, csv_context,
                context_doc_count=context_doc_count,
            )
            logger.info(f"[TIMING] gemini_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < 150:
            logger.warning("LLM failed, using extractive answer")
            facts = extract_key_facts(merged_docs, question)
            logger.info(f"Extracted {len(facts)} facts")
            answer = generate_extractive_answer(facts, question)

        if not answer or len(answer) < 50:
            answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

        sources = format_sources(merged_docs)
        logger.info(f"[TIMING] TOTAL ask_question: {time.time() - t_start:.2f}s | {len(sources)} sources")

        result = {"answer": answer, "sources": sources}
        _response_cache.put(question, result)

        if return_formatted:
            return format_answer_output(answer, sources)

        return result

    except Exception as e:
        error_msg = f"Query error: {str(e)}"
        logger.error(error_msg)
        if return_formatted:
            return f"Error: {error_msg}"
        return {"answer": error_msg, "sources": [], "error": str(e)}


async def ask_question_async(question: str, return_formatted: bool = False, use_llm: bool = True) -> Dict:
    """
    Async version of ask_question for FastAPI concurrent request handling.

    Uses asyncio.to_thread for sync I/O operations (Qdrant, embeddings) and
    Gemini async LLM calls. This allows multiple user requests
    to be processed concurrently without blocking the event loop.
    """
    t_start = time.time()
    health_status = await asyncio.to_thread(check_qdrant_health)
    logger.info(f"[TIMING] async health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        error_message = f"""
Database Error: {health_status['message']}

Details: {health_status.get('details', 'No additional details')}
Action: {health_status.get('action', 'Contact administrator')}

Error Type: {health_status['error']}
"""
        if return_formatted:
            return error_message
        return {
            "answer": error_message,
            "sources": [],
            "error": health_status
        }

    logger.info("Database is healthy - proceeding with async query")

    # --- RESPONSE CACHE LOOKUP ---
    cached = _response_cache.get(question)
    if cached is not None:
        logger.info(f"[TIMING] TOTAL ask_question_async (CACHED): {time.time() - t_start:.2f}s")
        if return_formatted:
            return format_answer_output(cached["answer"], cached["sources"])
        return cached
    # --- END CACHE LOOKUP ---

    if CSV_PATH.exists():
        try:
            logger.info("Checking author query...")
            is_handled, csv_response = await asyncio.to_thread(
                handle_author_query, question, str(CSV_PATH)
            )

            if is_handled and csv_response and csv_response.strip():
                logger.info("CSV query matched - passing to LLM for summarization")
                t0 = time.time()
                csv_content = _build_csv_user_content(question, csv_response)
                llm_summary = await generate_llm_answer_async(
                    question, context="", csv_context="",
                    user_content=csv_content, system_prompt=_CSV_SYSTEM_PROMPT,
                    disable_thinking=True,
                )
                logger.info(f"[TIMING] async gemini_llm (csv): {time.time() - t0:.2f}s")
                logger.info(f"CSV LLM gist length: {len(llm_summary or '')} chars")

                if not llm_summary or len(llm_summary) < 5:
                    llm_summary = ""

                # Combine: LLM gist + raw CSV data appended
                combined_answer = _combine_csv_answer(llm_summary, csv_response)

                result = {
                    "answer": combined_answer,
                    "sources": [],
                    "query_type": "author_csv",
                }
                _response_cache.put(question, result)

                if return_formatted:
                    return format_answer_output(combined_answer, [])
                return result
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    try:
        client = await asyncio.to_thread(get_qdrant_client)

        logger.info(f"Searching: {question[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = await asyncio.to_thread(
            searcher.search, question, 200, SCORE_THRESHOLD
        )
        logger.info(f"[TIMING] async hybrid_search: {time.time() - t0:.2f}s ({len(results)} results)")

        if not results:
            answer = "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        t0 = time.time()
        merged_docs = await asyncio.to_thread(
            merge_consecutive_chunks, client, results
        )
        logger.info(f"[TIMING] async merge_chunks: {time.time() - t0:.2f}s ({len(merged_docs)} docs)")

        if not merged_docs:
            answer = "போதுமான தகவல்கள் இல்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        # Select relevant docs (same set used for context + evidence)
        relevant_docs = _select_relevant_docs(merged_docs)
        logger.info(f"Selected {len(relevant_docs)} relevant documents for context")

        # Build context with equal excerpts from all relevant docs
        context, context_doc_count = build_context_from_docs(relevant_docs, question)

        # CSV semantic context
        csv_results = await asyncio.to_thread(search_csv_semantic, question, 3)
        logger.info("------ CSV Rows Sent To LLM (async) ------")
        for row in csv_results:
            logger.info(row)
        logger.info("------------------------------------------")

        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])

        answer = ""
        if use_llm:
            t0 = time.time()
            answer = await generate_llm_answer_async(
                question, context, csv_context,
                context_doc_count=context_doc_count,
            )
            logger.info(f"[TIMING] async gemini_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < 100:
            logger.warning("LLM failed, using extractive answer")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)

        if not answer or len(answer) < 50:
            answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

        sources = format_sources(merged_docs)
        logger.info(f"[TIMING] TOTAL ask_question_async: {time.time() - t_start:.2f}s | {len(sources)} sources")

        result = {"answer": answer, "sources": sources}
        _response_cache.put(question, result)

        if return_formatted:
            return format_answer_output(answer, sources)

        return result

    except Exception as e:
        error_msg = f"Query error: {str(e)}"
        logger.error(error_msg)
        if return_formatted:
            return f"Error: {error_msg}"
        return {"answer": error_msg, "sources": [], "error": str(e)}


def ask_question_stream(question: str):
    """
    Streaming version of ask_question.
    Yields dicts: {"type": "token", "content": str} for answer tokens,
    and {"type": "sources", "sources": list} at the end.
    For non-streamable responses (CSV queries, errors), yields complete answer as single token.
    """
    # 1. Check health
    health_status = check_qdrant_health()
    if not health_status["healthy"]:
        yield {"type": "token", "content": f"Database Error: {health_status['message']}"}
        yield {"type": "sources", "sources": []}
        return

    # --- RESPONSE CACHE LOOKUP ---
    cached = _response_cache.get(question)
    if cached is not None:
        logger.info("[CACHE] Streaming cache hit - yielding full cached answer")
        yield {"type": "token", "content": cached["answer"]}
        yield {"type": "sources", "sources": cached["sources"]}
        return
    # --- END CACHE LOOKUP ---

    # 2. Check CSV queries first — stream LLM summary of CSV data
    if CSV_PATH.exists():
        try:
            is_handled, csv_response = handle_author_query(question, str(CSV_PATH))
            if is_handled and csv_response and csv_response.strip():
                logger.info("CSV query matched - streaming LLM summary")
                csv_content = _build_csv_user_content(question, csv_response)
                accumulated = []
                for token in generate_llm_answer_stream(
                    question, context="", csv_context="",
                    user_content=csv_content, system_prompt=_CSV_SYSTEM_PROMPT,
                    disable_thinking=True,
                ):
                    accumulated.append(token)
                    yield {"type": "token", "content": token}

                llm_summary = "".join(accumulated)
                logger.info(f"CSV streaming gist length: {len(llm_summary)} chars")
                if len(llm_summary) < 5:
                    # Streaming failed — try sync fallback
                    logger.warning("CSV streaming gist too short, trying sync fallback")
                    llm_summary = generate_llm_answer(
                        question, context="", csv_context="",
                        user_content=csv_content, system_prompt=_CSV_SYSTEM_PROMPT,
                        disable_thinking=True,
                    )
                    if llm_summary and len(llm_summary) >= 5:
                        yield {"type": "token", "content": llm_summary}
                    else:
                        fallback = "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:"
                        yield {"type": "token", "content": fallback}
                        llm_summary = ""

                # Append raw CSV data separator + data after the streamed summary
                csv_suffix = _csv_data_suffix(csv_response)
                yield {"type": "token", "content": csv_suffix}

                combined_answer = _combine_csv_answer(llm_summary, csv_response)
                _response_cache.put(question, {"answer": combined_answer, "sources": []})
                yield {"type": "sources", "sources": []}
                return
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    # 3. Vector search + LLM streaming
    try:
        client = get_qdrant_client()

        logger.info(f"Streaming search: {question[:60]}...")
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=200, score_threshold=SCORE_THRESHOLD)

        if not results:
            yield {"type": "token", "content": "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."}
            yield {"type": "sources", "sources": []}
            return

        merged_docs = merge_consecutive_chunks(client, results)
        if not merged_docs:
            yield {"type": "token", "content": "போதுமான தகவல்கள் இல்லை."}
            yield {"type": "sources", "sources": []}
            return

        logger.info(f"Merged into {len(merged_docs)} documents")

        # Select relevant docs (same set used for context + evidence)
        relevant_docs = _select_relevant_docs(merged_docs)
        logger.info(f"Selected {len(relevant_docs)} relevant documents for context")

        # Build context with equal excerpts from all relevant docs
        context, context_doc_count = build_context_from_docs(relevant_docs, question)

        csv_results = search_csv_semantic(question, top_k=3)
        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])

        # Stream LLM tokens and accumulate for caching
        token_count = 0
        accumulated_tokens = []
        for token in generate_llm_answer_stream(
            question, context, csv_context,
            context_doc_count=context_doc_count,
        ):
            token_count += 1
            accumulated_tokens.append(token)
            yield {"type": "token", "content": token}

        # If streaming produced too few tokens, fall back to extractive
        if token_count < 10:
            logger.warning("Streaming produced too few tokens, using extractive fallback")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)
            if answer:
                accumulated_tokens = [answer]
                yield {"type": "token", "content": answer}

        sources = format_sources(merged_docs)

        # --- CACHE STORE ---
        full_answer = "".join(accumulated_tokens)
        if full_answer and len(full_answer) >= 50:
            _response_cache.put(question, {"answer": full_answer, "sources": sources})
        # --- END CACHE STORE ---

        yield {"type": "sources", "sources": sources}

    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        yield {"type": "token", "content": f"Query error: {str(e)}"}
        yield {"type": "sources", "sources": []}


def preload_models():
    """
    Preload all models at app startup with visible spinners.
    Call this ONCE in your Streamlit app's initialization section.
    After this runs, all subsequent queries will be fast and silent.
    """
    logger.info("=" * 60)
    logger.info("PRELOADING MODELS FOR STREAMLIT")
    logger.info("=" * 60)

    with st.spinner("Loading embedding model..."):
        _ = get_embed_model()
        st.success("Embedding model loaded")

    with st.spinner("Connecting to Qdrant database..."):
        _ = get_qdrant_client()
        st.success("Qdrant connected")

    with st.spinner("Validating Gemini API..."):
        validate_gemini_api()
        st.success("Gemini API validated")

    logger.info("=" * 60)
    logger.info("ALL MODELS READY - APP IS READY TO SERVE")
    logger.info("=" * 60)

    st.success("All models loaded successfully! Ready to answer queries.")


# ============================================================================
# RE-EXPORTS FOR BACKWARD COMPATIBILITY
# All names that were previously available on `hybrid_search` continue to
# resolve here so that `import hybrid_search as hs; hs.X` and
# `from hybrid_search import X` keep working for every consumer.
# ============================================================================

from tamil_text import (  # noqa: E402, F401
    FUZZY_THRESHOLD,
    TYPO_DISTANCE,
    fuzzy_match_score,
    _edit_distance_one,
    _strip_tamil_possessive_suffixes,
    _strip_tamil_possessive_suffix_word,
    _PatternBank,
    _RE_INITIALS_NAME,
    _RE_TAMIL_WORD,
)

from csv_queries import (  # noqa: E402, F401
    normalize_author_name,
    flexible_author_match,
    _find_closest_author,
    _find_closest_title,
    EnhancedAuthorQuerySystem,
    format_author_list,
    format_author_topics,
    format_topic_authors,
    detect_issue_count_query,
    get_issue_count,
    format_issue_count,
    detect_start_year_query,
    get_start_year,
    handle_author_query,
    _csv_source,
    _csv_data_suffix,
    _combine_csv_answer,
    _author_system_cache,
    _author_system_lock,
)

from llm import (  # noqa: E402, F401
    PONNI_ABOUT_CONTEXT,
    TAMIL_ANSWER_SYSTEM_PROMPT,
    _CSV_SYSTEM_PROMPT,
    _get_gemini_client,
    _gemini_generation_config,
    _build_user_content,
    _build_csv_user_content,
    _WH_PATTERNS,
    _is_wh_question,
    _YES_NO_PATTERNS,
    _is_yes_no_question,
    generate_llm_answer,
    generate_llm_answer_async,
    generate_llm_answer_stream,
    generate_extractive_answer,
    _truncate_at_sentence_boundary,
    validate_gemini_api,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

from search import (  # noqa: E402, F401
    retrieve_all_chunks_for_document,
    merge_consecutive_chunks,
    extract_key_facts,
    _select_relevant_docs,
    _extract_relevant_excerpt,
    build_context_from_docs,
    format_sources,
    format_answer_output,
)
