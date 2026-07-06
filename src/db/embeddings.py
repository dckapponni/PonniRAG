"""Embedding, model loading, and hybrid search for Tamil documents.

Provides dense (E5) and sparse (BM25) embeddings, thread-safe singleton
loaders for the embedding model and Qdrant client, CSV semantic search,
and a hybrid Qdrant search class that fuses both vector types via RRF.

Usage::

    from embeddings import HybridQdrantSearch, get_qdrant_client
    client = get_qdrant_client()
    results = HybridQdrantSearch(client).search(query)
"""

import logging
import os
import unicodedata
from pathlib import Path
from typing import Dict, List

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

torch.set_grad_enabled(False)
import threading  # noqa: E402

import pandas as pd  # noqa: E402
from qdrant_client import models  # noqa: E402
from retry import with_qdrant_retry  # noqa: E402

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = "qdrant_indexer"
RAG_DEBUG = os.environ.get("RAG_DEBUG", "0").lower() in ("1", "true", "yes")

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CSV_PATH = BASE_DIR / "data" / "summary.csv"
SCORE_THRESHOLD = 0.65

_embed_lock = threading.Lock()

logger = logging.getLogger(__name__)


# HELPER — flatten author_name for embedding text
def _flatten_author(author_val) -> str:
    """Convert an author_name value to a plain display string.

    Handles lists (joined with ", "), bracket-wrapped CSV strings, and
    NA/NaN sentinels. Returns an empty string for empty or sentinel values.

    Args:
        author_val: Raw author value — list, str, or other.

    Returns:
        Clean author string suitable for embedding or display.
    """
    if isinstance(author_val, list):
        return ", ".join(str(a) for a in author_val if a and str(a).upper() != "NA")
    raw = str(author_val).strip()
    # Strip outer brackets if present (CSV format)
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    if raw.upper() in {"NA", "NAN", "NONE", ""}:
        return ""
    return raw


# THREAD-SAFE SINGLETON LOADERS
_singletons = {}
_singleton_locks = {
    "embed_model": threading.Lock(),
    "qdrant_client": threading.Lock(),
    "csv_dataframe": threading.Lock(),
    "csv_embeddings": threading.Lock(),
}


def _clear_singletons():
    """Clear all cached singletons. For use in tests only."""
    _singletons.clear()


def get_embed_model():
    """Return the multilingual E5 embedding model (thread-safe singleton).

    Loads the model on first call and caches it for all subsequent calls.
    Model is placed on CUDA if available, otherwise CPU.

    Returns:
        Loaded SentenceTransformer instance.
    """
    if "embed_model" not in _singletons:
        with _singleton_locks["embed_model"]:
            if "embed_model" not in _singletons:
                logger.info(
                    f"Loading embedding model on {DEVICE} (this happens only once)..."
                )
                model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
                logger.info(f"Embedding model loaded on {DEVICE} and cached")
                _singletons["embed_model"] = model
    return _singletons["embed_model"]


def get_qdrant_client() -> QdrantClient:
    """Return a connected Qdrant client (thread-safe singleton).

    Connects to the server at QDRANT_HOST:QDRANT_PORT on first call
    and verifies the target collection exists.

    Returns:
        Connected QdrantClient instance.

    Raises:
        Exception: If the server is unreachable or the collection is missing.
    """
    if "qdrant_client" not in _singletons:
        with _singleton_locks["qdrant_client"]:
            if "qdrant_client" not in _singletons:
                logger.info(f"Using Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
                collection_info = client.get_collection(COLLECTION_NAME)
                logger.info(f"Connected: {collection_info.points_count} points")
                _singletons["qdrant_client"] = client
    return _singletons["qdrant_client"]


def get_csv_dataframe():
    """Return the article summary CSV as a DataFrame (thread-safe singleton).

    Loads and caches the CSV on first call. Supports multi-author bracket
    fields. Returns an empty DataFrame if the file is missing or unreadable.

    Returns:
        Loaded DataFrame, or empty DataFrame on failure.
    """
    if "csv_dataframe" not in _singletons:
        with _singleton_locks["csv_dataframe"]:
            if "csv_dataframe" not in _singletons:
                if not CSV_PATH.exists():
                    _singletons["csv_dataframe"] = pd.DataFrame()
                else:
                    try:
                        df = pd.read_csv(
                            CSV_PATH,
                            encoding="utf-8",
                            engine="python",
                            quotechar='"',
                            skipinitialspace=True,
                            on_bad_lines="warn",
                        )
                        df.columns = df.columns.str.strip()
                        logger.info(f"CSV loaded safely: {len(df)} rows")
                    except Exception as e:
                        logger.error(f"CSV loading failed: {e}")
                        df = pd.DataFrame()

                    _singletons["csv_dataframe"] = df
    return _singletons["csv_dataframe"]


def get_csv_embeddings():
    """Return precomputed embeddings for all CSV rows (thread-safe singleton).

    Builds embedding text per row by joining all column values, flattening
    author fields via _flatten_author. Encodes with the E5 model on first
    call and caches the result.

    Returns:
        List of (text, embedding) tuples, or empty list if CSV is missing.
    """
    if "csv_embeddings" not in _singletons:
        with _singleton_locks["csv_embeddings"]:
            if "csv_embeddings" not in _singletons:
                df = get_csv_dataframe()
                model = get_embed_model()

                if df.empty:
                    _singletons["csv_embeddings"] = []
                else:
                    # Identify the author column (Tamil header)
                    author_col = next((c for c in df.columns if "ஆசிரியர்" in c), None)

                    texts = []
                    for _, row in df.iterrows():
                        parts = []
                        for col in df.columns:
                            val = row[col]
                            if not pd.notna(val):
                                continue
                            # Flatten author field — strip brackets, join names
                            if col == author_col:
                                flat = _flatten_author(val)
                                if flat:
                                    parts.append(flat)
                            else:
                                parts.append(str(val))
                        texts.append(" | ".join(parts))

                    embeddings = model.encode(
                        [f"passage: {t}" for t in texts],
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    _singletons["csv_embeddings"] = list(zip(texts, embeddings))
                    logger.info(f"CSV embeddings computed: {len(texts)} rows")

    return _singletons["csv_embeddings"]


def dense_embed_query(text: str):
    """Encode a query string into a normalised dense embedding vector.

    Applies NFC Unicode normalisation and prepends the ``query:`` prefix
    required by the E5 model.

    Args:
        text: Query string to encode.

    Returns:
        Normalised embedding as a Python list of floats.
    """
    model = get_embed_model()
    text = unicodedata.normalize("NFC", text)
    with _embed_lock:
        return model.encode(
            f"query: {text}",
            normalize_embeddings=True,
        ).tolist()


def sparse_embed(text: str):
    """Generate a sparse BM25 embedding for a search query.

    Delegates to the shared fastembed ``Qdrant/bm25`` encoder so that
    indexer and query-time tokenisation are identical.

    Args:
        text: Query string to encode.

    Returns:
        Sparse vector compatible with Qdrant sparse search.
    """
    from sparse import sparse_embed_query

    return sparse_embed_query(text)


def search_csv_semantic(question: str, top_k: int = 5):
    """Search CSV rows by semantic similarity to a query.

    Encodes the query and computes dot-product similarity against all
    precomputed CSV row embeddings.

    Args:
        question: Query string.
        top_k: Number of top results to return (default 5).

    Returns:
        List of up to top_k matching row text strings, ranked by score.
    """
    csv_data = get_csv_embeddings()
    model = get_embed_model()

    if not csv_data:
        return []

    with _embed_lock:
        query_emb = model.encode(
            f"query: {question}",
            normalize_embeddings=True,
        )

    scored = []
    for text, emb in csv_data:
        score = float(torch.tensor(query_emb) @ torch.tensor(emb))
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scored[:top_k]]


# QDRANT HEALTH & HYBRID SEARCH
def check_qdrant_health() -> Dict:
    """Check Qdrant server connectivity and collection status.

    Returns:
        Dict with ``healthy`` (bool), ``points_count`` on success, or
        ``error`` and ``action`` keys on failure.
    """
    try:
        client = get_qdrant_client()
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            return {
                "healthy": True,
                "collection": COLLECTION_NAME,
                "points_count": collection_info.points_count,
                "message": "Qdrant server is healthy",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": "collection_not_found",
                "message": f"Collection '{COLLECTION_NAME}' not found",
                "details": str(e),
                "action": "Create the collection on the server",
            }
    except Exception as e:
        return {
            "healthy": False,
            "error": "connection_failed",
            "message": "Failed to connect to Qdrant server",
            "details": str(e),
            "action": "Check Qdrant server connection or run the indexer",
        }


class HybridQdrantSearch:
    """Hybrid vector search over the Qdrant collection.

    Fuses dense (E5) and sparse (BM25) retrieval using Reciprocal Rank
    Fusion (RRF). Optionally filters results by taxonomy tag IDs.
    Debug mode (RAG_DEBUG=1) logs per-branch hits to diagnose fusion quality.
    """

    def __init__(self, client: QdrantClient):
        """Initialise with a connected Qdrant client.

        Args:
            client: Active QdrantClient instance.
        """
        self.client = client

    def search(
        self,
        query: str,
        limit: int = 30,
        score_threshold: float = SCORE_THRESHOLD,
        tags: List[str] = None,
    ):
        """Run a hybrid dense + sparse search and return fused result points.

        Prefetches dense candidates (with score threshold) and sparse candidates
        separately, then fuses them with RRF. Optionally restricts results to
        articles bearing any of the supplied taxonomy tag IDs.

        Args:
            query: Search query string.
            limit: Maximum number of results to return (default 30).
            score_threshold: Minimum dense similarity score (default SCORE_THRESHOLD).
            tags: Optional list of taxonomy tag IDs to filter by.

        Returns:
            List of Qdrant ScoredPoint objects with payloads attached.
        """
        filter_conditions = [
            models.FieldCondition(key="type", match=models.MatchValue(value="article"))
        ]
        if tags:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.tags",
                    match=models.MatchAny(any=tags),
                )
            )
        search_filter = models.Filter(must=filter_conditions)

        dense_vec = dense_embed_query(query)
        sparse_vec = sparse_embed(query)

        if RAG_DEBUG:
            self._debug_branch_compare(
                query, dense_vec, sparse_vec, search_filter, score_threshold, limit
            )

        response = with_qdrant_retry(
            self.client.query_points,
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    filter=search_filter,
                    score_threshold=score_threshold,
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    filter=search_filter,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _debug_branch_compare(
        self, query, dense_vec, sparse_vec, search_filter, score_threshold, limit
    ):
        """Log dense-only and sparse-only top hits to diagnose RRF fusion.

        Issues two extra Qdrant queries per call. Only enabled when
        RAG_DEBUG=1 — never use in production.

        Args:
            query: Original query string (logged for context).
            dense_vec: Dense embedding vector.
            sparse_vec: Sparse BM25 vector.
            search_filter: Qdrant filter applied to both branches.
            score_threshold: Dense score threshold.
            limit: Result limit passed to the main search.
        """
        try:
            dense_resp = with_qdrant_retry(
                self.client.query_points,
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                using="dense",
                query_filter=search_filter,
                score_threshold=score_threshold,
                limit=10,
                with_payload=True,
            )
            sparse_resp = with_qdrant_retry(
                self.client.query_points,
                collection_name=COLLECTION_NAME,
                query=sparse_vec,
                using="sparse",
                query_filter=search_filter,
                limit=10,
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"[RAG_DEBUG] branch_compare failed: {e}")
            return

        dense_pts = dense_resp.points
        sparse_pts = sparse_resp.points

        def _heading(p):
            pl = p.payload or {}
            meta = pl.get("metadata", {}) or {}
            return (meta.get("heading") or pl.get("heading") or "")[:60]

        logger.info(f"[RAG_DEBUG] query={query[:80]!r}")
        logger.info(
            f"[RAG_DEBUG] dense_hits={len(dense_pts)} sparse_hits={len(sparse_pts)}"
        )

        sparse_token_count = (
            len(sparse_vec.indices) if hasattr(sparse_vec, "indices") else 0
        )
        nonzero_sparse = (
            sum(1 for v in sparse_vec.values if v > 0)
            if hasattr(sparse_vec, "values")
            else 0
        )
        logger.info(
            f"[RAG_DEBUG] sparse_tokens={sparse_token_count} nonzero={nonzero_sparse}"
        )

        logger.info("[RAG_DEBUG] --- dense top-10 ---")
        for i, p in enumerate(dense_pts, 1):
            logger.info(
                f"[RAG_DEBUG] dense {i:2d}. id={p.id} score={p.score:.4f} "
                f"heading={_heading(p)!r}"
            )
        logger.info("[RAG_DEBUG] --- sparse top-10 ---")
        for i, p in enumerate(sparse_pts, 1):
            logger.info(
                f"[RAG_DEBUG] sparse {i:2d}. id={p.id} score={p.score:.4f} "
                f"heading={_heading(p)!r}"
            )

        dense_ids = {p.id for p in dense_pts}
        sparse_ids = {p.id for p in sparse_pts}
        overlap = dense_ids & sparse_ids
        logger.info(
            f"[RAG_DEBUG] top10_overlap={len(overlap)}/10 "
            f"dense_only={len(dense_ids - sparse_ids)} "
            f"sparse_only={len(sparse_ids - dense_ids)}"
        )
