"""Cross-encoder reranker for hybrid-search sources.

Reorders top-K retrieved sources by a query-document relevance score
from a multilingual cross-encoder. Replaces lexical token-overlap
reranking with a learned semantic signal that handles Tamil
morphology, paraphrase, and short content words natively.

Fail-safe: if model load or inference raises, callers fall back to
the legacy lexical filter. Controlled by env flags:

    ENABLE_RERANKER   "1"/"true" (default) — set to "0" to disable.
    RERANKER_MODEL    HF model id (default: BAAI/bge-reranker-v2-m3).
    RERANKER_TOP_IN   candidates fed to reranker (default: 30).
    RERANKER_TOP_OUT  results kept after reranking (default: 10).
    RERANKER_MAX_CHARS per-doc content truncation (default: 800).
"""

import logging
import os
import threading
import unicodedata
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

ENABLE_RERANKER = os.environ.get("ENABLE_RERANKER", "1").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_IN = int(os.environ.get("RERANKER_TOP_IN", "30"))
RERANKER_TOP_OUT = int(os.environ.get("RERANKER_TOP_OUT", "10"))
RERANKER_MAX_CHARS = int(os.environ.get("RERANKER_MAX_CHARS", "800"))
RERANKER_BATCH = int(os.environ.get("RERANKER_BATCH", "16"))

_model = None
_model_lock = threading.Lock()
_load_failed = False


def is_enabled() -> bool:
    """Return True when the reranker is enabled and has not failed to load."""
    return ENABLE_RERANKER and not _load_failed


def get_reranker():
    """Load cross-encoder on first use. Returns None on failure.

    Subsequent calls after a load failure short-circuit to None so
    callers can fall back to lexical reranking without retrying.
    """
    global _model, _load_failed
    if _load_failed:
        return None
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL}")
            _model = CrossEncoder(RERANKER_MODEL, max_length=512)
            logger.info("Cross-encoder loaded")
        except Exception as e:
            _load_failed = True
            logger.warning(
                f"Cross-encoder load failed ({e!r}); reranker disabled this session"
            )
            return None
    return _model


def _doc_text(src: Dict) -> str:
    """Build the text passed to the cross-encoder for a single source."""
    parts = [
        str(src.get("heading") or ""),
        str(src.get("content") or src.get("text") or ""),
    ]
    text = " ".join(p for p in parts if p).strip()
    text = unicodedata.normalize("NFC", text)
    return text[:RERANKER_MAX_CHARS]


def rerank_sources(
    question: str,
    sources: List[Dict],
    top_in: int = None,
    top_out: int = None,
) -> Tuple[List[Dict], bool]:
    """Rerank sources with cross-encoder.

    Args:
        question: User query.
        sources: List of source dicts (already formatted by format_sources).
        top_in: Candidate pool size (default RERANKER_TOP_IN).
        top_out: Results kept (default RERANKER_TOP_OUT).

    Returns:
        (reranked_sources, success). When success is False the caller
        should fall back to the legacy lexical filter. The returned
        sources list is empty on failure.
    """
    if not sources:
        return [], True
    if not is_enabled():
        return [], False

    model = get_reranker()
    if model is None:
        return [], False

    top_in = top_in or RERANKER_TOP_IN
    top_out = top_out or RERANKER_TOP_OUT

    candidates = sources[:top_in]
    question_norm = unicodedata.normalize("NFC", question)

    pairs = [(question_norm, _doc_text(s)) for s in candidates]

    try:
        scores = model.predict(
            pairs,
            batch_size=RERANKER_BATCH,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda t: float(t[1]), reverse=True)
        reranked = []
        for src, score in scored[:top_out]:
            out = dict(src)
            out["rerank_score"] = float(score)
            reranked.append(out)
    except Exception as e:
        logger.warning(f"Cross-encoder rerank failed ({e!r}); falling back")
        return [], False

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            f"[RERANK] {len(candidates)} -> {len(reranked)} "
            f"(model={RERANKER_MODEL})"
        )
        for i, item in enumerate(reranked[:5], 1):
            heading = (item.get("heading") or "")[:60]
            logger.info(
                f"[RERANK] {i:2d}. ce={item['rerank_score']:.4f} "
                f"heading={heading!r}"
            )

    return reranked, True
