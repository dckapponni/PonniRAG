"""
Hybrid Search Orchestrator for Tamil Document Processing.
Coordinates health checks, caching, CSV queries, vector search,
and LLM answer generation. Delegates model loading, embeddings,
and search to sub-modules (cache, embeddings, search, llm, etc.).
"""
from typing import List, Dict, Optional
import logging
import asyncio
import time

from config.config import MAX_QUERY_LENGTH

from guardrails import (
    sanitize_query, detect_injection, validate_history,
    safe_error_response, safe_error_message, sanitize_output,
    SAFE_ERROR_MESSAGE,
)
from cache import ResponseCache, _response_cache
from embeddings import (
    USE_CUDA, DEVICE, COLLECTION_NAME,
    EMBEDDING_MODEL, SCORE_THRESHOLD, BASE_DIR, CSV_PATH,
    _embed_lock, _clear_singletons,
    get_embed_model, get_qdrant_client, get_csv_dataframe, get_csv_embeddings,
    dense_embed_query, _deterministic_token_hash, sparse_embed, search_csv_semantic,
    check_qdrant_health, HybridQdrantSearch,
)

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

def ask_question(question: str, return_formatted: bool = False, use_llm: bool = True, filter_tags: List[str] = None, history: List[Dict] = None) -> Dict:
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
    question = truncate_query(question)
    history = validate_history(history)

    is_injection, severity = detect_injection(question)
    if is_injection and severity == "high":
        refusal = "மன்னிக்கவும், இந்தக் கேள்விக்கு பதிலளிக்க இயலவில்லை. பொன்னி இதழ் தொடர்பான கேள்விகளை கேளுங்கள்."
        if return_formatted:
            return refusal
        return {"answer": refusal, "sources": []}

    t_start = time.time()
    health_status = check_qdrant_health()
    logger.info(f"[TIMING] health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        logger.error(f"Database unhealthy: {health_status}")
        error_message = SAFE_ERROR_MESSAGE
        if return_formatted:
            return error_message
        return {
            "answer": error_message,
            "sources": [],
            "error": "database_unavailable"
        }

    logger.info("Database is healthy - proceeding with query")
    _response_cache.check_version(health_status.get("points_count"))

    # --- RESPONSE CACHE LOOKUP (skip when conversation history is present) ---
    if not history:
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
                gemini_health = check_gemini_health()
                if not gemini_health["healthy"]:
                    logger.error(f"Gemini unhealthy (csv path): {gemini_health}")
                    llm_summary = ""
                else:
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

        logger.info(f"Searching: {question[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=200, score_threshold=SCORE_THRESHOLD, tags=filter_tags)
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
            gemini_health = check_gemini_health()
            if not gemini_health["healthy"]:
                logger.error(f"Gemini unhealthy: {gemini_health}")
            else:
                t0 = time.time()
                answer = generate_llm_answer(
                    question, context, csv_context,
                    context_doc_count=context_doc_count,
                    history=history,
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


async def ask_question_async(question: str, return_formatted: bool = False, use_llm: bool = True, filter_tags: List[str] = None, history: List[Dict] = None) -> Dict:
    """
    Async version of ask_question for FastAPI concurrent request handling.

    Uses asyncio.to_thread for sync I/O operations (Qdrant, embeddings) and
    Gemini async LLM calls. This allows multiple user requests
    to be processed concurrently without blocking the event loop.
    """
    question = truncate_query(question)
    history = validate_history(history)

    is_injection, severity = detect_injection(question)
    if is_injection and severity == "high":
        refusal = "மன்னிக்கவும், இந்தக் கேள்விக்கு பதிலளிக்க இயலவில்லை. பொன்னி இதழ் தொடர்பான கேள்விகளை கேளுங்கள்."
        if return_formatted:
            return refusal
        return {"answer": refusal, "sources": []}

    t_start = time.time()
    health_status = await asyncio.to_thread(check_qdrant_health)
    logger.info(f"[TIMING] async health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        logger.error(f"Database unhealthy: {health_status}")
        error_message = SAFE_ERROR_MESSAGE
        if return_formatted:
            return error_message
        return {
            "answer": error_message,
            "sources": [],
            "error": "database_unavailable"
        }

    logger.info("Database is healthy - proceeding with async query")
    _response_cache.check_version(health_status.get("points_count"))

    # --- RESPONSE CACHE LOOKUP (skip when conversation history is present) ---
    if not history:
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
                gemini_health = await asyncio.to_thread(check_gemini_health)
                if not gemini_health["healthy"]:
                    logger.error(f"Gemini unhealthy (async csv path): {gemini_health}")
                    llm_summary = ""
                else:
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
                if not history:
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
            searcher.search, question, 200, SCORE_THRESHOLD, filter_tags
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
            gemini_health = await asyncio.to_thread(check_gemini_health)
            if not gemini_health["healthy"]:
                logger.error(f"Gemini unhealthy (async): {gemini_health}")
            else:
                t0 = time.time()
                answer = await generate_llm_answer_async(
                    question, context, csv_context,
                    context_doc_count=context_doc_count,
                    history=history,
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


def ask_question_stream(question: str, filter_tags: List[str] = None, history: List[Dict] = None):
    """
    Streaming version of ask_question.
    Yields dicts: {"type": "token", "content": str} for answer tokens,
    and {"type": "sources", "sources": list} at the end.
    For non-streamable responses (CSV queries, errors), yields complete answer as single token.
    """
    question = truncate_query(question)
    history = validate_history(history)

    is_injection, severity = detect_injection(question)
    if is_injection and severity == "high":
        yield {"type": "token", "content": "மன்னிக்கவும், இந்தக் கேள்விக்கு பதிலளிக்க இயலவில்லை. பொன்னி இதழ் தொடர்பான கேள்விகளை கேளுங்கள்."}
        yield {"type": "sources", "sources": []}
        return

    # 1. Check health
    health_status = check_qdrant_health()
    if not health_status["healthy"]:
        logger.error(f"Database unhealthy: {health_status}")
        yield {"type": "token", "content": SAFE_ERROR_MESSAGE}
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
                    fallback = "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:"
                    yield {"type": "token", "content": fallback}
                    csv_suffix = _csv_data_suffix(csv_response)
                    yield {"type": "token", "content": csv_suffix}
                    yield {"type": "sources", "sources": []}
                    return
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
                if not history:
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
        results = searcher.search(question, limit=200, score_threshold=SCORE_THRESHOLD, tags=filter_tags)

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

        # Pre-check Gemini health before streaming LLM
        gemini_health = check_gemini_health()
        if not gemini_health["healthy"]:
            logger.error(f"Gemini unhealthy (stream): {gemini_health}")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)
            if answer:
                yield {"type": "token", "content": answer}
            else:
                yield {"type": "token", "content": "மன்னிக்கவும், LLM சேவை தற்போது கிடைக்கவில்லை."}
            yield {"type": "sources", "sources": format_sources(merged_docs)}
            return

        # Stream LLM tokens and accumulate for caching
        token_count = 0
        accumulated_tokens = []
        for token in generate_llm_answer_stream(
            question, context, csv_context,
            context_doc_count=context_doc_count,
            history=history,
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

        # --- CACHE STORE (skip when conversation history is present) ---
        full_answer = "".join(accumulated_tokens)
        full_answer = sanitize_output(full_answer)
        if not history and full_answer and len(full_answer) >= 50:
            _response_cache.put(question, {"answer": full_answer, "sources": sources})
        # --- END CACHE STORE ---

        yield {"type": "sources", "sources": sources}

    except Exception as e:
        logger.error(f"Streaming query error: {e}", exc_info=True)
        yield {"type": "token", "content": SAFE_ERROR_MESSAGE}
        yield {"type": "sources", "sources": []}


# ============================================================================
# DEPRECATED RE-EXPORTS — Backward compatibility only.
# New code should import from the source modules directly:
#   embeddings, csv_queries, llm, search, tamil_text, cache
# These re-exports exist because test_hybrid_search.py uses `hs.X` for ~50+
# names. Do not add new names here.
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
    _build_multi_turn_contents,
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
    check_gemini_health,
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
