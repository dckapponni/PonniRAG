"""
Embedding, model loading, and hybrid search for Tamil document processing.
Provides dense/sparse embeddings, Qdrant client management, CSV semantic search,
health checks, and hybrid vector search combining dense + sparse vectors.
"""
from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re
import os
import logging
import hashlib
from pathlib import Path
import streamlit as st

import torch
torch.set_grad_enabled(False)
from qdrant_client import models
import pandas as pd
import threading

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

logger = logging.getLogger(__name__)


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

    def search(self, query: str, limit: int = 30, score_threshold: float = SCORE_THRESHOLD, tags: List[str] = None):
        """
        Perform hybrid search using dense and sparse vectors.

        Executes a two-stage search combining dense embeddings for semantic similarity
        and sparse embeddings for keyword matching, then fuses results using RRF.

        Args:
            query (str): Search query string in Tamil or English
            limit (int, optional): Maximum number of results to return. Defaults to 30.
            score_threshold (float, optional): Minimum cosine similarity for dense
                vector results. Defaults to SCORE_THRESHOLD (0.8).
            tags (List[str], optional): Filter results to articles matching any of these tag IDs.

        Returns:
            List[ScoredPoint]: List of scored points from Qdrant with fused relevance scores
        """
        filter_conditions = [
            models.FieldCondition(key="type", match=models.MatchValue(value="article"))
        ]
        if tags:
            filter_conditions.append(
                models.FieldCondition(key="metadata.tags", match=models.MatchAny(any=tags))
            )
        search_filter = models.Filter(must=filter_conditions)

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_embed_query(query),
                    using="dense",
                    filter=search_filter,
                    score_threshold=score_threshold,
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=sparse_embed(query),
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
