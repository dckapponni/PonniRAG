"""
Hybrid Search Orchestrator for Tamil Document Processing.
Coordinates health checks, caching, CSV queries, vector search,
and LLM answer generation. Delegates model loading, embeddings,
and search to sub-modules (cache, embeddings, search, llm, etc.).
"""
from typing import List, Dict, Optional
import re
import os
import logging
from pathlib import Path
import asyncio
import time

from cache import ResponseCache, _response_cache 
from embeddings import (
    USE_CUDA, DEVICE, COLLECTION_NAME,
    EMBEDDING_MODEL, SCORE_THRESHOLD, BASE_DIR, CSV_PATH,
    _embed_lock,
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


# ============================================================================
# ORCHESTRATOR FUNCTIONS
# ============================================================================

def ask_question(question: str, return_formatted: bool = False, use_llm: bool = True, filter_tags: List[str] = None) -> Dict:
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


async def ask_question_async(question: str, return_formatted: bool = False, use_llm: bool = True, filter_tags: List[str] = None) -> Dict:
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


def ask_question_stream(question: str, filter_tags: List[str] = None):
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
