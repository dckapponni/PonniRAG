"""Hybrid Search Orchestrator for Tamil Document Processing.

Coordinates health checks, caching, CSV queries, vector search,
and LLM answer generation. Delegates model loading, embeddings,
and search to sub-modules (cache, embeddings, search, llm, etc.).
"""

import asyncio
import logging
import re as _re
import time
from typing import Dict, List

from cache import _response_cache
from csv_queries import (  # noqa: F401 — re-exported for backward compatibility
    EnhancedAuthorQuerySystem,
    _author_system_cache,
    _author_system_lock,
    _combine_csv_answer,
    _csv_data_suffix,
    _load_csv_safe,
    correct_query_spelling,
    detect_issue_count_query,
    flexible_author_match,
    format_author_list,
    format_author_topics,
    format_issue_count,
    format_topic_authors,
    get_issue_count,
    handle_author_query,
    normalize_author_name,
)
from embeddings import (  # noqa: F401 — re-exported for backward compatibility
    CSV_PATH,
    SCORE_THRESHOLD,
    HybridQdrantSearch,
    _embed_lock,
    check_qdrant_health,
    dense_embed_query,
    get_embed_model,
    get_qdrant_client,
    search_csv_semantic,
    sparse_embed,
)
from guardrails import (
    SAFE_ERROR_MESSAGE,
    detect_injection,
    safe_error_message,
    safe_error_response,
    sanitize_output,
    sanitize_query,
    validate_history,
)
from llm import _build_csv_user_content  # ✅ required
from llm import _get_csv_system_prompt  # ✅ required
from llm import (
    check_gemini_health,
    generate_extractive_answer,
    generate_llm_answer,
    generate_llm_answer_async,
    generate_llm_answer_stream,
)
from search import _select_relevant_docs  # ✅ required
from search import retrieve_all_chunks_for_document  # noqa: F401 — re-export
from search import (
    build_context_from_docs,
    extract_key_facts,
    format_answer_output,
    format_sources,
    merge_consecutive_chunks,
)

from config.config import MAX_QUERY_LENGTH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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


_LLM_MIN_ANSWER_LENGTH = 150  # Minimum chars for an LLM answer to be considered valid


def _msg(language: str, ta: str, en: str) -> str:
    """Return the appropriate message based on language."""
    return en if language == "en" else ta


def _llm_fallback_answer(question: str, merged_docs, history, language: str = "ta"):
    """Try cache then extractive fallback when LLM answer is insufficient.

    Called when the LLM returns an empty or too-short answer (rate limit,
    server error, timeout). Tries the response cache first (even in
    history mode — a stale LLM answer is better than extractive), then
    falls back to keyword-based extractive answer.

    Returns:
        (answer, fallback_reason): answer string and reason tag for the
        response metadata ("cached_response" or "extractive").
    """
    # Try cache — this helps in history mode where cache was skipped at the top
    cached = _response_cache.get(question)
    if (
        cached
        and cached.get("answer")
        and len(cached["answer"]) >= _LLM_MIN_ANSWER_LENGTH
    ):
        logger.info("[FALLBACK] Using cached response (LLM unavailable)")
        return cached["answer"], "cached_response"

    # Fall back to extractive
    logger.info("[FALLBACK] Using extractive answer (LLM unavailable)")
    facts = extract_key_facts(merged_docs, question)
    answer = generate_extractive_answer(facts, question)

    if not answer or len(answer) < 50:
        answer = _msg(
            language,
            "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன.",
            "The information for this question is available in the sources.",
        )

    return answer, "extractive"


# Patterns indicating the LLM answer says "data not available / not indexed".
# When detected, sources should be suppressed because they are partial keyword
# matches that don't actually answer the user's question.
_NO_DATA_PATTERNS = [
    # Tamil — explicit "not available / not in database" phrases
    _re.compile(r"தகவல்\s*(கள்\s*)?கிடைக்கவில்லை"),
    _re.compile(r"தகவல்\s*(கள்\s*)?இல்லை"),
    _re.compile(r"தரவுத்தளத்தில்\s*(தற்போது\s*)?இல்லை"),
    _re.compile(r"பதிவு\s*செய்யப்படவில்லை"),
    _re.compile(r"இன்னும்\s*(பதிவு|குறியீடு|நுழைவு)"),
    _re.compile(r"(indexed|index)\s*செய்யப்படவில்லை"),
    _re.compile(r"குறிப்பிட்ட\s*(இதழ்|தொகுதி|கட்டுரை).*இல்லை"),
    _re.compile(r"(இந்த|குறிப்பிட்ட).*தரவு.*இல்லை"),
    # Tamil — broader "not found / unavailable" phrases
    _re.compile(
        r"தற்போது\s*(இந்த|இதற்கான)?\s*தகவல்"
    ),  # "currently this information..."
    _re.compile(r"கிடைக்கவில்லை"),  # "not available" (standalone)
    _re.compile(r"இடம்\s*பெற(வில்லை|ற்று\s*இல்லை)"),  # "not included / not featured"
    _re.compile(r"காணப்படவில்லை"),  # "not found"
    _re.compile(r"கண்டுபிடிக்க\s*(இயல|முடி)வில்லை"),  # "unable to find"
    _re.compile(r"தொடர்பான\s*தகவல்.*இல்லை"),  # "no info related to..."
    _re.compile(r"உள்ளடக்கத்தில்.*இல்லை"),  # "not in the content"
    _re.compile(r"சேகரிக்கப்படவில்லை"),  # "not collected"
    _re.compile(r"நேரடியாக\s*தொடர்பில்லை"),  # "not directly related"
    # English — original patterns
    _re.compile(r"not\s+(yet\s+)?(been\s+)?indexed", _re.I),
    _re.compile(r"data\s+is\s+not\s+(currently\s+)?available", _re.I),
    _re.compile(r"no\s+(relevant\s+)?information\s+(is\s+)?(available|found)", _re.I),
    _re.compile(
        r"not\s+(currently\s+)?(available|present)"
        r"\s+in\s+(the\s+)?(database|archive|collection)",
        _re.I,
    ),
    _re.compile(
        r"has\s+not\s+(yet\s+)?been\s+(digitized|processed|extracted|added)", _re.I
    ),
    _re.compile(
        r"(this|the)\s+(specific\s+)?(issue|volume|article|data)"
        r"\s+(is\s+)?not\s+(available|found|indexed)",
        _re.I,
    ),
    _re.compile(r"no\s+data\s+(is\s+)?(available|found)", _re.I),
    _re.compile(r"do(es)?\s+not\s+(currently\s+)?contain", _re.I),
    # English — broader patterns
    _re.compile(r"not\s+(currently\s+)?available\s+in\s+(the\s+)?database", _re.I),
    _re.compile(
        r"(could|cannot|can'?t)\s+(not\s+)?find\s+(any\s+)?(relevant|specific|direct)",
        _re.I,
    ),
    _re.compile(
        r"no\s+(specific|direct|relevant)\s+(information|data|content|mention)", _re.I
    ),
    _re.compile(
        r"do(es)?\s+not\s+(directly\s+)?(address|answer|contain|cover|mention)", _re.I
    ),
    _re.compile(
        r"(context|documents?)\s+(provided\s+)?do(es)?"
        r"\s+not\s+(directly\s+)?(relate|pertain|answer)",
        _re.I,
    ),
    _re.compile(r"not\s+directly\s+related\s+to", _re.I),
    _re.compile(r"unable\s+to\s+(find|locate|identify)", _re.I),
    _re.compile(
        r"(don'?t|do\s+not)\s+have\s+(any\s+)?"
        r"(information|data)\s+(about|on|regarding)",
        _re.I,
    ),
]


# Regex: at least one alphanumeric, Tamil, or CJK character
_HAS_MEANINGFUL_CONTENT = _re.compile(r"[\w\u0B80-\u0BFF]")


def _query_has_meaningful_content(question: str) -> bool:
    """Check if the query has a meaningful character."""
    return bool(_HAS_MEANINGFUL_CONTENT.search(question))


def _answer_indicates_no_data(answer: str) -> bool:
    """Check if the LLM answer indicates the requested data is not available."""
    if not answer:
        return False
    for pattern in _NO_DATA_PATTERNS:
        if pattern.search(answer):
            logger.info(f"[NO_DATA] Answer matches no-data pattern: {pattern.pattern}")
            return True
    return False


# Stop words excluded from relevance checks — these are too generic to be
# useful for deciding whether a document is truly relevant to a query.
_TAMIL_STOP_WORDS = {
    "இதழில்",
    "இதழ்",
    "பொன்னி",
    "பொன்னியில்",
    "என்ன",
    "யாவை",
    "யார்",
    "எனும்",
    "பற்றி",
    "பற்றிய",
    "என்று",
    "உள்ள",
    "உள்ளது",
    "இருக்கு",
    "முக்கிய",
    "முக்கியமான",
    "கருத்துக்கள்",
    "கருத்து",
    "தகவல்",
    "கட்டுரை",
    "கட்டுரைகள்",
    "எழுதிய",
    "எழுதியவர்",
    "ஆசிரியர்",
    "தொகுதி",
    "இருக்கிறது",
    "இருந்தது",
    "செய்த",
    "செய்யும்",
    "எப்படி",
    "எங்கே",
    "எப்போது",
    "ஏன்",
    "எவ்வாறு",
    "எத்தனை",
    "கூறுக",
    "விளக்குக",
    "விவரி",
    "பட்டியலிடுக",
    "சுருக்கமாக",
    "இருக்கிறார்",
    "இருக்கின்றன",
    "வெளிவந்தது",
    "வெளியான",
}
_ENGLISH_STOP_WORDS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "that",
    "this",
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "about",
    "from",
    "with",
    "into",
    "ponni",
    "magazine",
    "issue",
    "volume",
    "article",
    "written",
    "author",
    "list",
    "tell",
    "explain",
    "describe",
    "main",
    "key",
    "important",
    "topics",
    "content",
}


def _check_context_relevance(question: str, relevant_docs: list) -> bool:
    """Check if retrieved documents are actually relevant to the query.

    Extracts distinguishing entities (years, names, specific terms) from
    the question and verifies at least some appear in the retrieved docs.
    Returns False when documents are likely fuzzy/partial keyword matches
    that don't actually address the user's question.

    This is deliberately conservative — it only returns False when it is
    confident the docs are irrelevant (key entities completely missing).
    """
    if not relevant_docs:
        return False

    # --- Extract distinguishing terms from the query ---
    years = set(_re.findall(r"\b(19\d{2}|20\d{2})\b", question))
    tamil_terms = set(_re.findall(r"[\u0B80-\u0BFF]{3,}", question))
    tamil_terms -= _TAMIL_STOP_WORDS
    english_terms = {w.lower() for w in _re.findall(r"[a-zA-Z]{3,}", question)}
    english_terms -= _ENGLISH_STOP_WORDS

    # If the query has no distinguishing terms (generic questions like
    # "tell me about Ponni"), assume the context is relevant.
    if not years and not tamil_terms and not english_terms:
        return True

    # --- Build combined text from retrieved documents ---
    all_text_parts = []
    from embeddings import _flatten_author

    for doc in relevant_docs[:10]:
        all_text_parts.append(str(doc.get("content", "")))
        all_text_parts.append(str(doc.get("heading", "")))
        all_text_parts.append(str(doc.get("doc_issue", "")))
        all_text_parts.append(str(doc.get("volume", "")))

        author_val = doc.get("author_name", "")
        all_text_parts.append(_flatten_author(author_val))

    # ✅ FIX: Combine into single text
    all_text = " ".join(all_text_parts).lower()

    # --- Year check ---
    # If the query asks about a specific year, at least one doc must
    # reference that year in content or metadata.  A year match is strong
    # enough evidence of relevance on its own.
    if years:
        if any(y in all_text for y in years):
            return True
        logger.info(f"[RELEVANCE] Year(s) {years} not found in any document")
        return False

    # --- Key-term check ---
    # At least one distinguishing Tamil/English term from the query must
    # appear somewhere in the retrieved documents.
    # Tamil is agglutinative — suffixes change word endings
    # (e.g. சுராதா→சுராதாவின், கல்வெட்டு→கல்வெட்டின்) so we use
    # common-prefix matching: two Tamil words match if they share a
    # prefix that is ≥60% of the shorter word (min 3 chars).
    key_terms = tamil_terms | english_terms
    if key_terms:
        doc_tamil_words = set(_re.findall(r"[\u0B80-\u0BFF]{3,}", all_text))

        def _common_prefix_len(a, b):
            n = min(len(a), len(b))
            for i in range(n):
                if a[i] != b[i]:
                    return i
            return n

        def _term_found(term):
            tl = term.lower()
            # Direct substring in full text (works well for English)
            if tl in all_text:
                return True
            # Tamil stem match via common prefix
            for dw in doc_tamil_words:
                shorter = min(len(dw), len(term))
                if shorter < 3:
                    continue
                cp = _common_prefix_len(dw, term)
                if cp >= max(3, int(shorter * 0.6)):
                    return True
            return False

        matched = sum(1 for t in key_terms if _term_found(t))
        if matched == 0:
            logger.info(
                f"[RELEVANCE] No key terms matched. " f"Query terms: {key_terms}"
            )
            return False

    return True


def truncate_query(question: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Normalize Unicode to NFC and truncate to max_length at a word boundary.

    NFC normalization ensures visually identical Tamil text always has
    identical byte representation (composed form).  Truncation prevents
    embedding latency spikes (E5 tokenizer caps at 512 tokens) and
    wasted LLM prompt budget (question is injected twice).
    """
    from tamil_text import normalize_unicode

    question = normalize_unicode(question)
    question = sanitize_query(question)

    if len(question) <= max_length:
        return question

    truncated = question[:max_length]
    # Cut at last whitespace to avoid splitting a word/Tamil character
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]

    logger.warning(
        f"Query truncated from {len(question)} to {len(truncated)} chars "
        f"(limit {max_length})"
    )
    return truncated


# ============================================================================
# ORCHESTRATOR FUNCTIONS
# ============================================================================


def ask_question(
    question: str,
    return_formatted: bool = False,
    use_llm: bool = True,
    filter_tags: List[str] = None,
    history: List[Dict] = None,
    language: str = "ta",
) -> Dict:
    """Answer a question using hybrid search and LLM generation.

    Primary entry point for processing user queries. Handles
    database health checks, author queries, vector search,
    document merging, LLM generation, and source formatting.
    Results are filtered by score threshold rather than a fixed
    top_k count. CSV queries are checked BEFORE vector search
    to return direct data without LLM.

    Args:
        question: User's question in Tamil or English.
        return_formatted: If True, return formatted string;
            if False, return dict. Defaults to False.
        use_llm: If True, use LLM for answer generation;
            if False, use extractive fallback. Defaults True.
        filter_tags: Optional tag filter list.
        history: Conversation history.
        language: Response language code.

    Returns:
        dict or str depending on return_formatted.
    """
    question = truncate_query(question)
    history = validate_history(history)

    # Reject queries with no meaningful content (only special characters / punctuation)
    if not _query_has_meaningful_content(question):
        msg = _msg(
            language,
            "சரியான கேள்வியை உள்ளிடவும். எழுத்துக்கள் அல்லது எண்கள் தேவை.",
            "Please enter a valid question with letters or numbers.",
        )
        if return_formatted:
            return msg
        return {"answer": msg, "sources": []}

    is_injection, severity = detect_injection(question)
    if is_injection and severity == "high":
        refusal = _msg(
            language,
            "மன்னிக்கவும், இந்தக் கேள்விக்கு பதிலளிக்க இயலவில்லை. பொன்னி இதழ் தொடர்பான கேள்விகளை கேளுங்கள்.",  # noqa: E501
            "Sorry, this query cannot be processed. Please ask questions related to Ponni magazine.",  # noqa: E501
        )
        if return_formatted:
            return refusal
        return {"answer": refusal, "sources": []}

    t_start = time.time()
    health_status = check_qdrant_health()
    logger.info(f"[TIMING] health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        logger.error(f"Database unhealthy: {health_status}")
        error_message = _msg(language, SAFE_ERROR_MESSAGE, safe_error_message("en"))
        if return_formatted:
            return error_message
        return {"answer": error_message, "sources": [], "error": "database_unavailable"}

    logger.info("Database is healthy - proceeding with query")
    _response_cache.check_version(health_status.get("points_count"))

    # --- RESPONSE CACHE LOOKUP (skip when conversation history is present) ---
    if not history:
        cached = _response_cache.get(question)
        if cached is not None:
            logger.info(
                f"[TIMING] TOTAL ask_question (CACHED): {time.time() - t_start:.2f}s"
            )
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
                gemini_health = check_gemini_health()
                if not gemini_health["healthy"]:
                    logger.error(f"Gemini unhealthy (csv path): {gemini_health}")
                    llm_summary = ""
                else:
                    t0 = time.time()
                    csv_content = _build_csv_user_content(
                        question, csv_response, language=language
                    )
                    llm_summary = generate_llm_answer(
                        question,
                        context="",
                        csv_context="",
                        user_content=csv_content,
                        system_prompt=_get_csv_system_prompt(language),
                        disable_thinking=True,
                        language=language,
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
                if not history:
                    _response_cache.put(question, result)

                if return_formatted:
                    return format_answer_output(combined_answer, [])
                return result
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    # 3. IF NOT CSV QUERY, PROCEED WITH VECTOR SEARCH + LLM
    try:
        client = get_qdrant_client()

        # Correct misspelled title/author words before vector search
        search_query = correct_query_spelling(question, str(CSV_PATH))

        logger.info(f"Searching: {search_query[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = searcher.search(
            search_query, limit=200, score_threshold=SCORE_THRESHOLD, tags=filter_tags
        )
        logger.info(
            f"[TIMING] hybrid_search: {time.time() - t0:.2f}s ({len(results)} results)"
        )

        if not results:
            answer = _msg(
                language,
                "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை.",
                "Sorry, no information found.",
            )
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        t0 = time.time()
        merged_docs = merge_consecutive_chunks(client, results)
        logger.info(
            f"[TIMING] merge_chunks: {time.time() - t0:.2f}s ({len(merged_docs)} docs)"
        )

        if not merged_docs:
            answer = _msg(
                language,
                "போதுமான தகவல்கள் இல்லை.",
                "Insufficient information available.",
            )
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
            csv_context = "\n".join(
                [f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)]
            )

        answer = ""
        fallback_reason = None
        if use_llm:
            gemini_health = check_gemini_health()
            if not gemini_health["healthy"]:
                logger.error(f"Gemini unhealthy: {gemini_health}")
            else:
                t0 = time.time()
                answer = generate_llm_answer(
                    question,
                    context,
                    csv_context,
                    context_doc_count=context_doc_count,
                    history=history,
                    language=language,
                )
                logger.info(f"[TIMING] gemini_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < _LLM_MIN_ANSWER_LENGTH:
            logger.warning("LLM answer insufficient, trying fallback")
            answer, fallback_reason = _llm_fallback_answer(
                question, merged_docs, history, language
            )

        sources = format_sources(merged_docs)

        # Suppress sources when the answer indicates the data is not available
        if _answer_indicates_no_data(answer):
            logger.info(
                "[NO_DATA] Suppressing sources — answer indicates data not available"
            )
            sources = []

        # Suppress sources when retrieved docs don't match query's key entities
        if sources and not _check_context_relevance(question, relevant_docs):
            logger.info(
                "[RELEVANCE] Suppressing sources — documents not relevant to query"
            )
            sources = []

        elapsed = time.time() - t_start
        logger.info(
            f"[TIMING] TOTAL ask_question: {elapsed:.2f}s" f" | {len(sources)} sources"
        )

        result = {"answer": answer, "sources": sources}
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        if not history:
            _response_cache.put(question, result)

        if return_formatted:
            return format_answer_output(answer, sources)

        return result

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        if return_formatted:
            return SAFE_ERROR_MESSAGE
        return safe_error_response()


async def ask_question_async(
    question: str,
    return_formatted: bool = False,
    use_llm: bool = True,
    filter_tags: List[str] = None,
    history: List[Dict] = None,
    language: str = "ta",
) -> Dict:
    """Answer a question asynchronously for FastAPI.

    Uses asyncio.to_thread for sync I/O operations (Qdrant,
    embeddings) and Gemini async LLM calls. This allows multiple
    user requests to be processed concurrently without blocking
    the event loop.
    """
    question = truncate_query(question)
    history = validate_history(history)

    # Reject queries with no meaningful content (only special characters / punctuation)
    if not _query_has_meaningful_content(question):
        msg = _msg(
            language,
            "சரியான கேள்வியை உள்ளிடவும். எழுத்துக்கள் அல்லது எண்கள் தேவை.",
            "Please enter a valid question with letters or numbers.",
        )
        if return_formatted:
            return msg
        return {"answer": msg, "sources": []}

    is_injection, severity = detect_injection(question)
    if is_injection and severity == "high":
        refusal = _msg(
            language,
            "மன்னிக்கவும், இந்தக் கேள்விக்கு பதிலளிக்க இயலவில்லை. பொன்னி இதழ் தொடர்பான கேள்விகளை கேளுங்கள்.",  # noqa: E501
            "Sorry, this query cannot be processed. Please ask questions related to Ponni magazine.",  # noqa: E501
        )
        if return_formatted:
            return refusal
        return {"answer": refusal, "sources": []}

    t_start = time.time()
    health_status = await asyncio.to_thread(check_qdrant_health)
    logger.info(f"[TIMING] async health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        logger.error(f"Database unhealthy: {health_status}")
        error_message = _msg(language, SAFE_ERROR_MESSAGE, safe_error_message("en"))
        if return_formatted:
            return error_message
        return {"answer": error_message, "sources": [], "error": "database_unavailable"}

    logger.info("Database is healthy - proceeding with async query")
    _response_cache.check_version(health_status.get("points_count"))

    # --- RESPONSE CACHE LOOKUP (skip when conversation history is present) ---
    if not history:
        cached = _response_cache.get(question)
        if cached is not None:
            elapsed = time.time() - t_start
            logger.info(
                "[TIMING] TOTAL ask_question_async" f" (CACHED): {elapsed:.2f}s"
            )
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
                gemini_health = await asyncio.to_thread(check_gemini_health)
                if not gemini_health["healthy"]:
                    logger.error(f"Gemini unhealthy (async csv path): {gemini_health}")
                    llm_summary = ""
                else:
                    t0 = time.time()
                    csv_content = _build_csv_user_content(
                        question, csv_response, language=language
                    )
                    llm_summary = await generate_llm_answer_async(
                        question,
                        context="",
                        csv_context="",
                        user_content=csv_content,
                        system_prompt=_get_csv_system_prompt(language),
                        disable_thinking=True,
                        language=language,
                    )
                    logger.info(
                        f"[TIMING] async gemini_llm (csv): {time.time() - t0:.2f}s"
                    )
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
                if not history:
                    _response_cache.put(question, result)

                if return_formatted:
                    return format_answer_output(combined_answer, [])
                return result
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    try:
        client = await asyncio.to_thread(get_qdrant_client)

        # Correct misspelled title/author words before vector search
        search_query = await asyncio.to_thread(
            correct_query_spelling, question, str(CSV_PATH)
        )

        logger.info(f"Searching: {search_query[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = await asyncio.to_thread(
            searcher.search, search_query, 200, SCORE_THRESHOLD, filter_tags
        )
        logger.info(
            f"[TIMING] async hybrid_search: "
            f"{time.time() - t0:.2f}s ({len(results)} results)"
        )

        if not results:
            answer = _msg(
                language,
                "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை.",
                "Sorry, no information found.",
            )
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        t0 = time.time()
        merged_docs = await asyncio.to_thread(merge_consecutive_chunks, client, results)
        logger.info(
            f"[TIMING] async merge_chunks: "
            f"{time.time() - t0:.2f}s ({len(merged_docs)} docs)"
        )

        if not merged_docs:
            answer = _msg(
                language,
                "போதுமான தகவல்கள் இல்லை.",
                "Insufficient information available.",
            )
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
            csv_context = "\n".join(
                [f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)]
            )

        answer = ""
        fallback_reason = None
        if use_llm:
            gemini_health = await asyncio.to_thread(check_gemini_health)
            if not gemini_health["healthy"]:
                logger.error(f"Gemini unhealthy (async): {gemini_health}")
            else:
                t0 = time.time()
                answer = await generate_llm_answer_async(
                    question,
                    context,
                    csv_context,
                    context_doc_count=context_doc_count,
                    history=history,
                    language=language,
                )
                logger.info(f"[TIMING] async gemini_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < _LLM_MIN_ANSWER_LENGTH:
            logger.warning("LLM answer insufficient (async), trying fallback")
            answer, fallback_reason = _llm_fallback_answer(
                question, merged_docs, history, language
            )

        sources = format_sources(merged_docs)

        # Suppress sources when the answer indicates the data is not available
        if _answer_indicates_no_data(answer):
            logger.info(
                "[NO_DATA] Suppressing sources — answer "
                "indicates data not available (async)"
            )
            sources = []

        # Suppress sources when retrieved docs don't match query's key entities
        if sources and not _check_context_relevance(question, relevant_docs):
            logger.info(
                "[RELEVANCE] Suppressing sources — "
                "documents not relevant to query (async)"
            )
            sources = []

        elapsed = time.time() - t_start
        logger.info(
            f"[TIMING] TOTAL ask_question_async: "
            f"{elapsed:.2f}s | {len(sources)} sources"
        )

        result = {"answer": answer, "sources": sources}
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        if not history:
            _response_cache.put(question, result)

        if return_formatted:
            return format_answer_output(answer, sources)

        return result

    except Exception as e:
        logger.error(f"Async query error: {e}", exc_info=True)
        if return_formatted:
            return SAFE_ERROR_MESSAGE
        return safe_error_response()


def ask_question_stream(
    question: str,
    filter_tags: List[str] = None,
    history: List[Dict] = None,
    language: str = "ta",
):
    """Stream answer tokens for a question.

    Yields dicts: {"type": "token", "content": str} for answer
    tokens, and {"type": "sources", "sources": list} at the end.
    For non-streamable responses (CSV queries, errors), yields
    complete answer as single token.
    """
    question = truncate_query(question)
    history = validate_history(history)

    # Reject queries with no meaningful content (only special characters / punctuation)
    if not _query_has_meaningful_content(question):
        yield {
            "type": "token",
            "content": _msg(
                language,
                "சரியான கேள்வியை உள்ளிடவும். எழுத்துக்கள் அல்லது எண்கள் தேவை.",
                "Please enter a valid question with letters or numbers.",
            ),
        }
        yield {"type": "sources", "sources": []}
        return

    is_injection, severity = detect_injection(question)
    if is_injection and severity == "high":
        yield {
            "type": "token",
            "content": _msg(
                language,
                "மன்னிக்கவும், இந்தக் கேள்விக்கு பதிலளிக்க இயலவில்லை. பொன்னி இதழ் தொடர்பான கேள்விகளை கேளுங்கள்.",  # noqa: E501
                "Sorry, this query cannot be processed. Please ask questions related to Ponni magazine.",  # noqa: E501
            ),
        }
        yield {"type": "sources", "sources": []}
        return

    # 1. Check health
    health_status = check_qdrant_health()
    if not health_status["healthy"]:
        logger.error(f"Database unhealthy: {health_status}")
        yield {
            "type": "token",
            "content": _msg(language, SAFE_ERROR_MESSAGE, safe_error_message("en")),
        }
        yield {"type": "sources", "sources": []}
        return

    _response_cache.check_version(health_status.get("points_count"))

    # --- RESPONSE CACHE LOOKUP (skip when conversation history is present) ---
    if not history:
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
                gemini_health = check_gemini_health()
                if not gemini_health["healthy"]:
                    logger.error(f"Gemini unhealthy (stream csv path): {gemini_health}")
                    fallback = _msg(
                        language,
                        "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:",
                        "Information retrieved from article database:",
                    )
                    yield {"type": "token", "content": fallback}
                    csv_suffix = _csv_data_suffix(csv_response)
                    yield {"type": "token", "content": csv_suffix}
                    yield {"type": "sources", "sources": []}
                    return
                csv_content = _build_csv_user_content(
                    question, csv_response, language=language
                )
                csv_sys_prompt = _get_csv_system_prompt(language)
                accumulated = []
                for token in generate_llm_answer_stream(
                    question,
                    context="",
                    csv_context="",
                    user_content=csv_content,
                    system_prompt=csv_sys_prompt,
                    disable_thinking=True,
                    language=language,
                ):
                    accumulated.append(token)
                    yield {"type": "token", "content": token}

                llm_summary = "".join(accumulated)
                logger.info(f"CSV streaming gist length: {len(llm_summary)} chars")
                if len(llm_summary) < 5:
                    # Streaming failed — try sync fallback
                    logger.warning("CSV streaming gist too short, trying sync fallback")
                    llm_summary = generate_llm_answer(
                        question,
                        context="",
                        csv_context="",
                        user_content=csv_content,
                        system_prompt=csv_sys_prompt,
                        disable_thinking=True,
                        language=language,
                    )
                    if llm_summary and len(llm_summary) >= 5:
                        yield {"type": "token", "content": llm_summary}
                    else:
                        fallback = _msg(
                            language,
                            "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:",
                            "Information retrieved from article database:",
                        )
                        yield {"type": "token", "content": fallback}
                        llm_summary = ""

                # Append raw CSV data separator + data after the streamed summary
                csv_suffix = _csv_data_suffix(csv_response)
                yield {"type": "token", "content": csv_suffix}

                combined_answer = _combine_csv_answer(llm_summary, csv_response)
                if not history:
                    _response_cache.put(
                        question, {"answer": combined_answer, "sources": []}
                    )
                yield {"type": "sources", "sources": []}
                return
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    # 3. Vector search + LLM streaming
    try:
        client = get_qdrant_client()

        # Correct misspelled title/author words before vector search
        search_query = correct_query_spelling(question, str(CSV_PATH))

        logger.info(f"Streaming search: {search_query[:60]}...")
        searcher = HybridQdrantSearch(client)
        results = searcher.search(
            search_query, limit=200, score_threshold=SCORE_THRESHOLD, tags=filter_tags
        )

        if not results:
            yield {
                "type": "token",
                "content": _msg(
                    language,
                    "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை.",
                    "Sorry, no information found.",
                ),
            }
            yield {"type": "sources", "sources": []}
            return

        merged_docs = merge_consecutive_chunks(client, results)
        if not merged_docs:
            yield {
                "type": "token",
                "content": _msg(
                    language,
                    "போதுமான தகவல்கள் இல்லை.",
                    "Insufficient information available.",
                ),
            }
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
            csv_context = "\n".join(
                [f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)]
            )

        # Pre-check Gemini health before streaming LLM
        gemini_health = check_gemini_health()
        if not gemini_health["healthy"]:
            logger.error(f"Gemini unhealthy (stream): {gemini_health}")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)
            if answer:
                yield {"type": "token", "content": answer}
            else:
                yield {
                    "type": "token",
                    "content": _msg(
                        language,
                        "மன்னிக்கவும், LLM சேவை தற்போது கிடைக்கவில்லை.",
                        "Sorry, the LLM service is currently unavailable.",
                    ),
                }
            yield {"type": "sources", "sources": format_sources(merged_docs)}
            return

        # Stream LLM tokens and accumulate for caching
        token_count = 0
        accumulated_tokens = []
        for token in generate_llm_answer_stream(
            question,
            context,
            csv_context,
            context_doc_count=context_doc_count,
            history=history,
            language=language,
        ):
            token_count += 1
            accumulated_tokens.append(token)
            yield {"type": "token", "content": token}

        # If streaming produced too few tokens, try cache then extractive
        fallback_reason = None
        if token_count < 10:
            logger.warning("Streaming produced too few tokens, trying fallback")
            answer, fallback_reason = _llm_fallback_answer(
                question, merged_docs, history, language
            )
            accumulated_tokens = [answer]
            yield {"type": "token", "content": answer}

        sources = format_sources(merged_docs)

        if fallback_reason:
            yield {"type": "fallback", "reason": fallback_reason}

        # --- CACHE STORE (skip when conversation history is present) ---
        full_answer = "".join(accumulated_tokens)
        full_answer = sanitize_output(full_answer)

        # Suppress sources when the answer indicates the data is not available
        if _answer_indicates_no_data(full_answer):
            logger.info(
                "[NO_DATA] Suppressing sources — answer "
                "indicates data not available (stream)"
            )
            sources = []

        # Suppress sources when retrieved docs don't match query's key entities
        if sources and not _check_context_relevance(question, relevant_docs):
            logger.info(
                "[RELEVANCE] Suppressing sources — "
                "documents not relevant to query (stream)"
            )
            sources = []

        if not history and full_answer and len(full_answer) >= 50:
            _response_cache.put(question, {"answer": full_answer, "sources": sources})
        # --- END CACHE STORE ---

        yield {"type": "sources", "sources": sources}

    except Exception as e:
        logger.error(f"Streaming query error: {e}", exc_info=True)
        yield {
            "type": "token",
            "content": _msg(language, SAFE_ERROR_MESSAGE, safe_error_message("en")),
        }
        yield {"type": "sources", "sources": []}
