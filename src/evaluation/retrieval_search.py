"""System wrappers for the retrieval ablation: hybrid vs dense vs sparse.

The production retriever (``HybridQdrantSearch``) fuses a dense E5 branch and a
sparse BM25 branch with Reciprocal Rank Fusion. To measure the *contribution*
of that fusion we need to run each branch in isolation against the same
collection, filter, and cut-off. This module exposes three callables with an
identical signature so the ablation driver can treat them uniformly:

    system(query: str, limit: int) -> List[str]   # ranked article keys

Each returns a de-duplicated list of *article keys* (not chunk points). A key
identifies one article across its chunks:

    key = "<volume> || <doc_issue> || <title>"

The same key function is used when building qrels, so ranked keys and judged
keys line up exactly.

Requires a running Qdrant with the indexed ``qdrant_indexer`` collection. Import
is lazy inside the constructor so importing this module in a unit-test
environment (no Qdrant) does not fail.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List

_EVAL_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EVAL_DIR.parent
_DB_DIR = _SRC_DIR / "db"
for _p in (str(_SRC_DIR), str(_DB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

# Only articles are retrievable evidence; mirrors the production filter.
_ARTICLE_TYPE = "article"


def article_key(payload: dict) -> str:
    """Build the stable article key from a Qdrant point payload.

    Groups all chunks of one article under a single key using the same
    ``(volume, doc_issue, title)`` tuple the production merger keys on.
    """
    meta = (payload or {}).get("metadata", {}) or {}
    volume = str(meta.get("volume", "")).strip()
    issue = str(meta.get("doc_issue", "")).strip()
    title = str(meta.get("title", "")).strip()
    return f"{volume} || {issue} || {title}"


def _dedup_keys(points) -> List[str]:
    """Collapse chunk-level points to article keys, preserving rank order."""
    seen = set()
    keys: List[str] = []
    for p in points:
        payload = p.payload or {}
        if payload.get("type") != _ARTICLE_TYPE:
            continue
        key = article_key(payload)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


class RetrievalSystems:
    """Factory holding a Qdrant client and the three ablation systems."""

    def __init__(self):
        """Connect to Qdrant and load the embedding encoders (lazy import)."""
        from embeddings import (  # noqa: PLC0415
            COLLECTION_NAME,
            SCORE_THRESHOLD,
            HybridQdrantSearch,
            dense_embed_query,
            get_qdrant_client,
            sparse_embed,
        )
        from qdrant_client import models  # noqa: PLC0415
        from retry import with_qdrant_retry  # noqa: PLC0415

        self._collection = COLLECTION_NAME
        self._score_threshold = SCORE_THRESHOLD
        self._models = models
        self._with_retry = with_qdrant_retry
        self._dense_embed = dense_embed_query
        self._sparse_embed = sparse_embed
        self.client = get_qdrant_client()
        self._hybrid = HybridQdrantSearch(self.client)

        self._article_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="type", match=models.MatchValue(value=_ARTICLE_TYPE)
                )
            ]
        )

    # -- individual systems -------------------------------------------------

    def hybrid(self, query: str, limit: int = 20) -> List[str]:
        """Production RRF fusion of the dense and sparse branches."""
        points = self._hybrid.search(
            query, limit=limit, score_threshold=self._score_threshold
        )
        return _dedup_keys(points)

    def dense(self, query: str, limit: int = 20) -> List[str]:
        """Dense-only E5 semantic retrieval (no BM25, no fusion)."""
        vec = self._dense_embed(query)
        resp = self._with_retry(
            self.client.query_points,
            collection_name=self._collection,
            query=vec,
            using="dense",
            query_filter=self._article_filter,
            score_threshold=self._score_threshold,
            limit=limit,
            with_payload=True,
        )
        return _dedup_keys(resp.points)

    def sparse(self, query: str, limit: int = 20) -> List[str]:
        """Sparse-only BM25 lexical retrieval (no dense, no fusion)."""
        vec = self._sparse_embed(query)
        resp = self._with_retry(
            self.client.query_points,
            collection_name=self._collection,
            query=vec,
            using="sparse",
            query_filter=self._article_filter,
            limit=limit,
            with_payload=True,
        )
        return _dedup_keys(resp.points)

    def as_dict(self) -> Dict[str, Callable[[str, int], List[str]]]:
        """Return the three systems keyed by the name used in the report."""
        return {"hybrid": self.hybrid, "dense": self.dense, "sparse": self.sparse}
