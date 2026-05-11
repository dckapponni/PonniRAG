"""Embedding, model loading, and hybrid search for Tamil documents.

Provides dense/sparse embeddings, Qdrant client management,
CSV semantic search, health checks, and hybrid vector search.
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


# ============================================================================
# HELPER — flatten author_name for embedding text
# ============================================================================


def _flatten_author(author_val) -> str:
    """Safely convert author_name to a plain string for embedding.

    author_name is now stored as a list in Qdrant metadata
    but the CSV column still holds the raw bracket string.

    Handles:
      list   -> join with ", "
      str    -> strip brackets and return
      other  -> str()
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


# ============================================================================
# THREAD-SAFE SINGLETON LOADERS
# ============================================================================

_singletons = {}
_singleton_locks = {
    "embed_model": threading.Lock(),
    "qdrant_client": threading.Lock(),
    "csv_dataframe": threading.Lock(),
    "csv_embeddings": threading.Lock(),
}


def _clear_singletons():
    """Clear all cached singletons. For testing only."""
    _singletons.clear()


def get_embed_model():
    """Load and cache embedding model (thread-safe singleton)."""
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
    """Qdrant server mode (thread-safe singleton)."""
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
    """Load CSV once and cache it (thread-safe singleton).

    Uses robust pandas parsing with no row loss.
    Supports multi-author fields with bracket-wrapped names.
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
    """Precompute embeddings for CSV rows (thread-safe singleton).

    Uses _flatten_author() to strip brackets and join author names
    before building embedding text for semantic search.
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
    """Encode a query string into a dense embedding vector."""
    model = get_embed_model()
    text = unicodedata.normalize("NFC", text)
    with _embed_lock:
        return model.encode(
            f"query: {text}",
            normalize_embeddings=True,
        ).tolist()


def sparse_embed(text: str):
    """Generate sparse BM25 embedding for a search query.

    Delegates to the shared fastembed ``Qdrant/bm25`` encoder so that
    indexer and runtime tokenize identically. The legacy tf-hash
    implementation diverged between index and query time, leaving the
    sparse branch effectively dead and the RRF fusion degenerate.
    """
    from sparse import sparse_embed_query

    return sparse_embed_query(text)


def search_csv_semantic(question: str, top_k: int = 5):
    """Semantic search over CSV rows."""
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
    """Hybrid search combining dense and sparse vectors for optimal results."""

    def __init__(self, client: QdrantClient):
        """Initialize with a Qdrant client instance."""
        self.client = client

    def search(
        self,
        query: str,
        limit: int = 30,
        score_threshold: float = SCORE_THRESHOLD,
        tags: List[str] = None,
    ):
        """Perform hybrid search using dense and sparse vectors."""
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

        Enabled via RAG_DEBUG=1. Issues two extra queries per call —
        do not enable in production.
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
