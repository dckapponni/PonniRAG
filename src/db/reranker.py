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
import re
import threading
import unicodedata
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

ENABLE_RERANKER = os.environ.get("ENABLE_RERANKER", "1").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_IN = int(os.environ.get("RERANKER_TOP_IN", "30"))
RERANKER_TOP_OUT = int(os.environ.get("RERANKER_TOP_OUT", "5"))
RERANKER_MIN_OUT = int(os.environ.get("RERANKER_MIN_OUT", "3"))
# After sorting by cross-encoder score, drop a candidate when its score
# falls below ``RERANKER_SCORE_FLOOR_RATIO`` × top_score *or* below
# ``RERANKER_GAP_RATIO`` × previous_score. Adaptive cut keeps the number
# of sources tight (typically 3–5) when only a few docs are clearly
# relevant, instead of padding to a fixed top-N with mediocre matches.
RERANKER_SCORE_FLOOR_RATIO = float(os.environ.get("RERANKER_SCORE_FLOOR_RATIO", "0.4"))
RERANKER_GAP_RATIO = float(os.environ.get("RERANKER_GAP_RATIO", "0.5"))
RERANKER_MAX_CHARS = int(os.environ.get("RERANKER_MAX_CHARS", "800"))
RERANKER_BATCH = int(os.environ.get("RERANKER_BATCH", "16"))
RERANKER_PIN_HEADING_MATCHES = os.environ.get(
    "RERANKER_PIN_HEADING_MATCHES", "1"
).lower() in ("1", "true", "yes")
# When heading-pin finds a high-confidence match, return pinned docs
# (plus a small filler) directly — without running the cross-encoder at
# all. Saves ~10-30s of CPU on title-style queries.
RERANKER_SKIP_CE_ON_CLEAR_WIN = os.environ.get(
    "RERANKER_SKIP_CE_ON_CLEAR_WIN", "1"
).lower() in ("1", "true", "yes")
# Minimum phrase length (in tokens) that counts as a "clear win" — a
# 3-token phrase in a heading is far less likely to be a coincidence
# than a 2-token one, but 2-token matches still skip CE if the pin
# returns very few matches.
RERANKER_CLEAR_WIN_MIN_PHRASE = int(
    os.environ.get("RERANKER_CLEAR_WIN_MIN_PHRASE", "2")
)
# Cap on number of pinned matches that still qualifies as "clear" — if
# many docs share the heading phrase, we still need CE to pick among them.
RERANKER_CLEAR_WIN_MAX_MATCHES = int(
    os.environ.get("RERANKER_CLEAR_WIN_MAX_MATCHES", "3")
)

# Minimum length of a Tamil/Latin token to be eligible for heading-phrase
# matching. Single-char tokens overmatch; very short ones tend to be
# inflectional fragments.
_HEADING_PHRASE_MIN_TOKEN = 2
# A "phrase" is N+ consecutive content tokens from the query appearing
# verbatim (separated by whitespace) in a candidate heading.
_HEADING_PHRASE_MIN_NGRAM = 2

_TAMIL_STOP = {
    "உள்ள",
    "என்ற",
    "என்று",
    "என்ன",
    "எந்த",
    "ஒரு",
    "இது",
    "அது",
    "இந்த",
    "அந்த",
    "மற்றும்",
    "பற்றி",
    "பற்றிய",
    "எனும்",
    "யாவை",
    "உள்ளது",
}

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


def _query_content_tokens(question: str) -> List[str]:
    """Extract ordered content tokens from a query, stripping stopwords."""
    q = unicodedata.normalize("NFC", question).lower()
    tokens = re.findall(r"[஀-௿a-z0-9]+", q)
    return [
        t
        for t in tokens
        if len(t) >= _HEADING_PHRASE_MIN_TOKEN and t not in _TAMIL_STOP
    ]


def _query_phrases(question: str) -> List[str]:
    """Build consecutive n-gram phrases from query content tokens.

    Yields longer phrases first so a literal title match like
    "மே தினம்" wins over a single-token coincidence.
    """
    toks = _query_content_tokens(question)
    phrases = []
    for n in range(min(4, len(toks)), _HEADING_PHRASE_MIN_NGRAM - 1, -1):
        for i in range(len(toks) - n + 1):
            phrases.append(" ".join(toks[i : i + n]))
    return phrases


def _clear_winner_matches(question: str, sources: List[Dict]) -> List[Dict]:
    """Return heading-phrase matches that justify skipping the cross-encoder.

    A match is "clear" when:
      - The matched query phrase has at least
        ``RERANKER_CLEAR_WIN_MIN_PHRASE`` content tokens.
      - The total number of matching docs is at most
        ``RERANKER_CLEAR_WIN_MAX_MATCHES`` (otherwise we need CE to
        pick among them).

    The returned list preserves original retrieval order. An empty
    list means there is no clear winner and the caller should run CE.
    """
    if not RERANKER_SKIP_CE_ON_CLEAR_WIN or not sources:
        return []

    phrases = _query_phrases(question)
    if not phrases:
        return []

    qualifying_phrases = [
        p for p in phrases if len(p.split()) >= RERANKER_CLEAR_WIN_MIN_PHRASE
    ]
    if not qualifying_phrases:
        return []

    matches: List[Dict] = []
    for src in sources:
        heading = unicodedata.normalize("NFC", str(src.get("heading") or "")).lower()
        if heading and any(p in heading for p in qualifying_phrases):
            matches.append(src)

    if not matches or len(matches) > RERANKER_CLEAR_WIN_MAX_MATCHES:
        return []
    return matches


def _pin_heading_matches(question: str, sources: List[Dict], top_in: int) -> List[Dict]:
    """Reorder ``sources`` so that heading-phrase matches lead.

    Any candidate whose heading contains a multi-token query phrase
    (with stopwords removed) is moved to the front, preserving the
    relative order among matches and among non-matches. Without this
    pin, a rank-68 candidate whose title literally equals the query
    phrase ("மே தினம்") never enters the cross-encoder's slice.
    """
    if not RERANKER_PIN_HEADING_MATCHES or not sources:
        return sources

    phrases = _query_phrases(question)
    if not phrases:
        return sources

    matches: List[Dict] = []
    rest: List[Dict] = []
    matched_idx_in_pool: List[int] = []

    for idx, src in enumerate(sources):
        heading = unicodedata.normalize("NFC", str(src.get("heading") or "")).lower()
        if heading and any(p in heading for p in phrases):
            matches.append(src)
            matched_idx_in_pool.append(idx)
        else:
            rest.append(src)

    if not matches:
        return sources

    logger.info(
        f"[RERANK] heading-phrase pin: {len(matches)} match(es) "
        f"raised from original ranks {matched_idx_in_pool[:10]}"
    )

    # Cap the lead block at top_in so we still rerank a healthy pool.
    head = matches[:top_in]
    tail_budget = max(0, top_in - len(head))
    return head + rest[:tail_budget] + matches[top_in:] + rest[tail_budget:]


def _adaptive_cutoff(scored: list, max_keep: int) -> List[Dict]:
    """Pick top sources using cross-encoder score gaps, not a fixed count.

    Always keeps at least ``RERANKER_MIN_OUT`` sources so the UI is never
    empty when the reranker did run. After the minimum is satisfied, a
    candidate is dropped when its score falls below either:

    * ``RERANKER_SCORE_FLOOR_RATIO`` × top_score — absolute relevance floor.
    * ``RERANKER_GAP_RATIO`` × previous_score — sharp drop-off between
      consecutive results.

    Together they yield a tight 3–5 source list when only a few docs are
    clearly relevant, and a fuller list (up to ``max_keep``) when several
    are roughly comparable.
    """
    if not scored:
        return []

    top_score = float(scored[0][1])
    floor = top_score * RERANKER_SCORE_FLOOR_RATIO
    min_keep = min(RERANKER_MIN_OUT, len(scored))
    cap = min(max_keep, len(scored))

    kept: List[Dict] = []
    prev_score = top_score
    for src, score in scored[:cap]:
        s = float(score)
        if len(kept) >= min_keep:
            if s < floor:
                break
            if prev_score > 0 and s < prev_score * RERANKER_GAP_RATIO:
                break
        out = dict(src)
        out["rerank_score"] = s
        kept.append(out)
        prev_score = s

    return kept


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

    top_in = top_in or RERANKER_TOP_IN
    top_out = top_out or RERANKER_TOP_OUT

    # Clear-winner short-circuit: avoid the cross-encoder entirely when
    # 1–N docs literally contain a multi-token query phrase in their
    # heading. Saves the full CE pass (~10-30s on CPU) for the typical
    # "give me article X" / "what does X say about Y" pattern.
    winners = _clear_winner_matches(question, sources)
    if winners:
        kept = winners[:top_out]
        result = []
        for src in kept:
            out = dict(src)
            out["rerank_score"] = 1.0  # sentinel — heading-pin bypass
            result.append(out)
        logger.info(
            f"[RERANK] clear-winner bypass: {len(result)} doc(s) returned "
            f"without cross-encoder"
        )
        return result, True

    model = get_reranker()
    if model is None:
        return [], False

    pool = _pin_heading_matches(question, sources, top_in)
    candidates = pool[:top_in]
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
        reranked = _adaptive_cutoff(scored, top_out)
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
