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
    """Return True when the reranker is enabled and has not permanently failed to load.

    Combines two conditions: the ``ENABLE_RERANKER`` environment flag must
    be truthy, and the module-level ``_load_failed`` flag must be ``False``
    (i.e. no previous :func:`get_reranker` call raised an exception).

    Returns:
        bool: ``True`` if the reranker may be used; ``False`` if it has been
        disabled via the environment flag or if the model failed to load
        during this process lifetime.
    """
    return ENABLE_RERANKER and not _load_failed


def get_reranker():
    """Load the cross-encoder model on first use and return the cached instance.

    Uses a double-checked locking pattern with ``_model_lock`` to ensure the
    model is initialized at most once in multi-threaded environments. After a
    load failure, ``_load_failed`` is set to ``True`` and all subsequent
    calls return ``None`` immediately without retrying, so callers can fall
    back to lexical reranking without incurring repeated import overhead.

    Returns:
        CrossEncoder | None: The loaded ``sentence_transformers.CrossEncoder``
        instance, or ``None`` if the model failed to load. A ``None`` return
        permanently disables the reranker for the current process.
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
    """Extract ordered content tokens from a query, stripping Tamil stop words.

    Applies Unicode NFC normalization, lowercases the question, and extracts
    all Tamil (``஀``–``௿``) and Latin alphanumeric token sequences. Tokens
    shorter than ``_HEADING_PHRASE_MIN_TOKEN`` characters or present in
    ``_TAMIL_STOP`` are discarded.

    Args:
        question (str): Raw user query string.

    Returns:
        List[str]: Ordered list of content token strings suitable for
        building query phrases for heading-match detection.
    """
    q = unicodedata.normalize("NFC", question).lower()
    tokens = re.findall(r"[஀-௿a-z0-9]+", q)
    return [
        t
        for t in tokens
        if len(t) >= _HEADING_PHRASE_MIN_TOKEN and t not in _TAMIL_STOP
    ]


def _query_phrases(question: str) -> List[str]:
    """Build consecutive n-gram phrases from a query's content tokens.

    Generates all contiguous n-grams of length ``_HEADING_PHRASE_MIN_NGRAM``
    up to ``min(4, n_tokens)`` from the content tokens produced by
    :func:`_query_content_tokens`. Longer phrases are listed first so that a
    literal multi-word title match (e.g. ``"மே தினம்"``) takes precedence
    over a single-token coincidence during heading comparison.

    Args:
        question (str): Raw user query string.

    Returns:
        List[str]: Ordered list of phrase strings, from longest to shortest
        n-gram, suitable for substring matching against document headings.
    """
    toks = _query_content_tokens(question)
    phrases = []
    for n in range(min(4, len(toks)), _HEADING_PHRASE_MIN_NGRAM - 1, -1):
        for i in range(len(toks) - n + 1):
            phrases.append(" ".join(toks[i : i + n]))
    return phrases


def _clear_winner_matches(question: str, sources: List[Dict]) -> List[Dict]:
    """Return heading-phrase matches that justify skipping the cross-encoder.

    Identifies sources whose heading literally contains a qualifying query
    phrase. A match qualifies as a "clear winner" and short-circuits the
    cross-encoder when both of the following hold:

    - The matched phrase has at least ``RERANKER_CLEAR_WIN_MIN_PHRASE``
      content tokens (default: 2), reducing false positives from
      single-token coincidences.
    - The total number of matching documents is at most
      ``RERANKER_CLEAR_WIN_MAX_MATCHES`` (default: 3); when many documents
      share a heading phrase the cross-encoder is still needed to rank them.

    When ``RERANKER_SKIP_CE_ON_CLEAR_WIN`` is disabled, always returns an
    empty list.

    Args:
        question (str): Raw user query string.
        sources (List[Dict]): Formatted source document dicts, each expected
            to contain a ``"heading"`` key.

    Returns:
        List[Dict]: The matching source dicts in their original retrieval
        order, or an empty list if no clear winner is found.
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
    """Reorder sources so that heading-phrase matches lead the candidate pool.

    Without this step, a highly relevant document whose heading literally
    matches a multi-token query phrase may rank beyond ``top_in`` in the
    initial retrieval order and never reach the cross-encoder. This function
    moves all such documents to the front of the list while preserving
    relative order within the matched and unmatched groups respectively.

    The lead block of matched documents is capped at ``top_in`` entries so
    the cross-encoder always receives a healthy and diverse candidate pool.
    Matched documents beyond the cap are appended after the non-matched
    remainder.

    When ``RERANKER_PIN_HEADING_MATCHES`` is disabled or no query phrases
    can be extracted, the original ``sources`` list is returned unchanged.

    Args:
        question (str): Raw user query string.
        sources (List[Dict]): Formatted source document dicts, each expected
            to contain a ``"heading"`` key.
        top_in (int): Maximum number of candidates to feed to the
            cross-encoder; used to size the pinned lead block.

    Returns:
        List[Dict]: Reordered source list with heading-phrase matches at the
        front, or the original list if no matches are found or the feature
        is disabled.
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
    """Select top sources from cross-encoder results using adaptive score-gap cutoffs.

    Rather than returning a fixed top-N, applies two data-driven stopping
    criteria after the mandatory minimum (``RERANKER_MIN_OUT``) is satisfied:

    - **Absolute floor**: drops a candidate when its score falls below
      ``RERANKER_SCORE_FLOOR_RATIO × top_score``, eliminating documents
      that are clearly less relevant than the best match.
    - **Gap ratio**: drops a candidate when its score falls below
      ``RERANKER_GAP_RATIO × previous_score``, catching sharp relevance
      drop-offs between consecutive ranked results.

    This yields a tight 3–5 source list when only a few documents are
    clearly relevant, and a fuller list (up to ``max_keep``) when several
    documents score comparably.

    Args:
        scored (list): List of ``(source_dict, score)`` tuples sorted by
            score in descending order, as produced by
            ``model.predict`` + ``sorted``.
        max_keep (int): Hard upper bound on the number of sources returned.

    Returns:
        List[Dict]: Source dicts that survived the cutoff, each augmented
        with a ``"rerank_score"`` (float) key. Returns an empty list if
        ``scored`` is empty.
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
    """Build the document text string passed to the cross-encoder for a single source.

    Concatenates the ``"heading"`` and ``"content"`` (or ``"text"``) fields
    of the source dict, separated by a space, after filtering out empty
    parts. Applies Unicode NFC normalization and truncates the result to
    ``RERANKER_MAX_CHARS`` characters to stay within the cross-encoder's
    token budget.

    Args:
        src (Dict): Formatted source document dict, expected to contain
            ``"heading"`` and optionally ``"content"`` or ``"text"`` keys.

    Returns:
        str: NFC-normalized, truncated document text string ready for
        pairing with the query in a cross-encoder ``predict`` call.
    """
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
    """Rerank retrieved sources using a multilingual cross-encoder model.

    Implements a three-path decision tree:

    1. **Clear-winner bypass** — if :func:`_clear_winner_matches` finds
       1–``RERANKER_CLEAR_WIN_MAX_MATCHES`` sources whose heading literally
       contains a multi-token query phrase, returns those sources immediately
       with a sentinel ``rerank_score`` of ``1.0``, skipping the
       cross-encoder entirely (saves 10–30 s of CPU on title-style queries).

    2. **Cross-encoder path** — pins heading-phrase matches to the front of
       the candidate pool via :func:`_pin_heading_matches`, slices to
       ``top_in`` candidates, calls ``model.predict`` in batches of
       ``RERANKER_BATCH``, sorts by descending score, and applies
       :func:`_adaptive_cutoff` to select the final ``top_out`` results.

    3. **Failure path** — if the model is not loaded, disabled, or raises
       during inference, returns ``([], False)`` so the caller falls back to
       the legacy lexical filter.

    Args:
        question (str): User query string.
        sources (List[Dict]): Formatted source document dicts produced by
            ``format_sources``, each expected to contain ``"heading"`` and
            ``"content"`` keys.
        top_in (int | None): Number of candidates fed to the cross-encoder.
            Defaults to ``RERANKER_TOP_IN`` (env: ``RERANKER_TOP_IN``,
            default 30).
        top_out (int | None): Maximum number of results kept after reranking.
            Defaults to ``RERANKER_TOP_OUT`` (env: ``RERANKER_TOP_OUT``,
            default 5).

    Returns:
        Tuple[List[Dict], bool]: A two-element tuple where the first element
        is the reranked (and possibly truncated) list of source dicts — each
        augmented with a ``"rerank_score"`` (float) key — and the second
        element is ``True`` on success or ``False`` when the caller should
        fall back to lexical filtering.
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
