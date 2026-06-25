"""Vector search and document processing for Tamil document retrieval.

Handles document chunk retrieval, merging, context building,
relevance filtering, and output formatting.
"""

import concurrent.futures
import hashlib
import logging
import re
from typing import Dict, List, Tuple

from embeddings import COLLECTION_NAME
from qdrant_client import QdrantClient, models
from retry import with_qdrant_retry

logger = logging.getLogger(__name__)

# Per-document chunk retrieval fans out one Qdrant scroll per candidate
# document. Running these serially is the dominant query latency (a 200-chunk
# candidate pool can map to >100 documents, each a ~350 ms round-trip). Hydrate
# them concurrently against the (thread-safe) HTTP client instead.
_HYDRATE_MAX_WORKERS = 16
_hydrate_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_HYDRATE_MAX_WORKERS, thread_name_prefix="hydrate"
)


def retrieve_all_chunks_for_document(
    client: QdrantClient,
    doc_id: str,
    doc_issue: str,
    volume: str,
    title: str = None,
) -> List[Dict]:
    """Retrieve all stored chunks for a specific article from Qdrant.

    Scrolls through the collection with a filter requiring ``type="article"``
    and matching ``metadata.doc_id``, ``metadata.doc_issue``, and
    ``metadata.volume``. When ``title`` is provided, an additional
    ``metadata.title`` filter is applied to isolate a single article within
    an issue; without it, all articles in the issue are returned (legacy
    behavior).

    Because ``doc_id`` in this schema identifies a magazine issue rather than
    an individual article, the ``title`` parameter is the primary mechanism
    for article-level isolation.

    Args:
        client (QdrantClient): Connected Qdrant client instance.
        doc_id (str): Magazine issue identifier stored in
            ``metadata.doc_id``.
        doc_issue (str): Issue label stored in ``metadata.doc_issue``.
        volume (str): Volume identifier stored in ``metadata.volume``.
        title (str | None): Article title to further restrict results to a
            single article. Pass ``None`` for issue-level retrieval.
            Defaults to ``None``.

    Returns:
        List[Dict]: List of Qdrant ``ScoredPoint`` or ``Record`` objects
        for all matching chunks, in the order returned by the scroll API.
        Returns an empty list if no chunks match the filter.
    """
    must = [
        models.FieldCondition(key="type", match=models.MatchValue(value="article")),
        models.FieldCondition(
            key="metadata.doc_id", match=models.MatchValue(value=doc_id)
        ),
        models.FieldCondition(
            key="metadata.doc_issue", match=models.MatchValue(value=doc_issue)
        ),
        models.FieldCondition(
            key="metadata.volume", match=models.MatchValue(value=volume)
        ),
    ]
    if title:
        must.append(
            models.FieldCondition(
                key="metadata.title", match=models.MatchValue(value=title)
            )
        )

    all_chunks = []
    offset = None
    while True:
        points, offset = with_qdrant_retry(
            client.scroll,
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=must),
            limit=100,
            offset=offset,
            with_payload=True,
        )
        all_chunks.extend(points)
        if offset is None:
            break
    return all_chunks


def merge_consecutive_chunks(client: QdrantClient, points) -> List[Dict]:
    """Merge all chunks belonging to the same article into a single document.

    Groups search result points by the composite key
    ``(volume, doc_id, doc_issue, title)`` — including title ensures that
    distinct articles within the same issue are merged independently rather
    than collapsing into a single 50 k-word issue blob. Points whose title
    is empty fall back to issue-level grouping (legacy behavior).

    For each unique document key, retrieves all chunks via
    :func:`retrieve_all_chunks_for_document`, sorts them by ``chunk_id``,
    and joins their content into a single whitespace-normalized string.
    Documents with fewer than 50 words after merging are discarded.

    The resulting list is sorted by the retrieval score of the first
    matching point (descending).

    Args:
        client (QdrantClient): Connected Qdrant client instance used for
            chunk retrieval.
        points: Iterable of Qdrant point objects returned by a hybrid search,
            each expected to carry a ``payload`` dict and a ``score`` field.

    Returns:
        List[Dict]: List of merged document dicts, each containing:
        ``"volume"``, ``"doc_id"``, ``"doc_issue"``, ``"heading"``,
        ``"author_name"``, ``"content"``, ``"word_count"``,
        ``"chunk_count"``, ``"score"``, and ``"tags"``.
        Sorted by ``"score"`` descending.
    """
    # Phase 1: collect unique candidate documents in score order (no I/O).
    seen_docs = set()
    candidates = []
    for p in points:
        payload = p.payload or {}
        if payload.get("type") != "article":
            continue

        metadata = payload.get("metadata", {})
        title = metadata.get("title", "") or ""
        doc_key = (
            metadata.get("volume"),
            metadata.get("doc_id"),
            metadata.get("doc_issue"),
            title,
        )

        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        candidates.append((p, metadata, title))

    # Phase 2: hydrate each candidate's full chunk set concurrently. Each
    # retrieval is an independent Qdrant scroll, so fanning them out turns
    # ~N serial round-trips into ~N / workers.
    def _hydrate(metadata, title):
        return retrieve_all_chunks_for_document(
            client,
            doc_id=metadata.get("doc_id"),
            doc_issue=metadata.get("doc_issue"),
            volume=metadata.get("volume"),
            title=title or None,
        )

    chunk_results = list(
        _hydrate_executor.map(
            lambda c: _hydrate(c[1], c[2]),
            candidates,
        )
    )

    # Phase 3: build merged documents in the original score order.
    merged_docs = []
    for (p, metadata, title), all_chunks in zip(candidates, chunk_results):
        if not all_chunks:
            continue

        chunk_data = []
        for chunk_point in all_chunks:
            chunk_payload = chunk_point.payload or {}
            chunk_data.append(
                {
                    "chunk_id": chunk_payload.get("chunk_id", 0),
                    "content": chunk_payload.get("content", "").strip(),
                }
            )

        chunk_data.sort(key=lambda x: x["chunk_id"])
        full_content = " ".join(chunk["content"] for chunk in chunk_data)
        full_content = re.sub(r"\s+", " ", full_content).strip()

        word_count = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", full_content))

        if word_count < 50:
            continue

        merged_docs.append(
            {
                "volume": metadata.get("doc_id", "unknown"),
                "doc_id": metadata.get("doc_id", "unknown"),
                "doc_issue": metadata.get("doc_issue", "unknown"),
                "heading": metadata.get("title", ""),
                "author_name": metadata.get("author_name", ""),
                "content": full_content,
                "word_count": word_count,
                "chunk_count": len(chunk_data),
                "score": p.score,
                "tags": metadata.get("tags", []),
            }
        )

    merged_docs.sort(key=lambda x: x["score"], reverse=True)
    return merged_docs


def extract_key_facts(docs: List[Dict], question: str) -> List[Dict]:
    """Extract sentences from documents that are most relevant to the question.

    Splits each document's content on sentence-ending punctuation (``।``,
    ``.``, ``!``, ``?``) and scores each sentence by counting occurrences
    of Tamil keyword tokens (≥ 2 characters) extracted from the question.
    Each keyword hit contributes 3 relevance points; only sentences scoring
    at least 5 points are kept. Used as input for
    :func:`generate_extractive_answer` when the LLM is unavailable.

    Args:
        docs (List[Dict]): List of merged document dicts, each containing
            a ``"content"`` key. At most the first 15 documents are examined.
        question (str): User question string; Tamil tokens extracted from it
            drive the relevance scoring.

    Returns:
        List[Dict]: Up to 20 fact dicts sorted by descending relevance score,
        each containing:

        - ``"sentence"`` (str): The extracted sentence.
        - ``"score"`` (int): Keyword-hit relevance score.
        - ``"source_issue"`` (str): ``doc_issue`` of the originating document.
        - ``"source_volume"`` (str): ``volume`` of the originating document.
    """
    facts = []
    q_keywords = set(re.findall(r"[\u0B80-\u0BFF]{2,}", question.lower()))

    for doc in docs[:15]:
        content = doc["content"]
        sentences = re.split(r"[.।!?]+", content)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30:
                continue

            sent_lower = sent.lower()
            relevance = 0

            for kw in q_keywords:
                if kw in sent_lower:
                    relevance += 3

            if relevance >= 5:
                facts.append(
                    {
                        "sentence": sent,
                        "score": relevance,
                        "source_issue": doc.get("doc_issue", "NA"),
                        "source_volume": doc.get("volume", "NA"),
                    }
                )

    facts.sort(key=lambda x: x["score"], reverse=True)
    return facts[:20]


def _select_relevant_docs(merged_docs: List[Dict]) -> List[Dict]:
    """Select relevant documents from a scored pool using adaptive score-gap filtering.

    Applies three progressive filtering layers, processing documents in
    descending score order:

    1. **Absolute floor** — drops any document scoring below
       ``35 %`` of the top document's score.
    2. **Gap detection** — stops when a document scores below ``40 %``
       of the immediately preceding document, catching sharp relevance
       drop-offs.
    3. **Diminishing-returns tightening** — after 20 documents are
       selected, the gap ratio tightens to ``50 %`` to prevent long
       tails of marginally relevant results.

    Additionally deduplicates documents by an MD5 hash of the first
    200 characters of their content before appending to the selection.
    Always returns at least 1 document (``MIN_SOURCES``) and at most
    100 (``MAX_SOURCES``).

    Args:
        merged_docs (List[Dict]): Pool of merged document dicts sorted by
            ``"score"`` descending, as produced by
            :func:`merge_consecutive_chunks`.

    Returns:
        List[Dict]: Filtered and deduplicated subset of ``merged_docs``,
        preserving the original score-descending order.
        Returns an empty list if ``merged_docs`` is empty.
    """
    MIN_SOURCES = 1
    MAX_SOURCES = 100
    FLOOR_RATIO = 0.35  # must score >= 35% of top doc
    GAP_RATIO = 0.4  # stop if doc scores < 40% of previous doc
    TIGHT_GAP_RATIO = 0.5  # tighter gap after TIGHT_GAP_AFTER docs
    TIGHT_GAP_AFTER = 20  # tighten gap ratio after this many docs

    if not merged_docs:
        return []

    top_score = merged_docs[0]["score"]
    floor_cutoff = top_score * FLOOR_RATIO

    selected = []
    seen_hashes = set()
    prev_score = top_score

    for doc in merged_docs:
        if len(selected) >= MAX_SOURCES:
            break

        score = doc["score"]

        # After minimum satisfied, check cutoffs
        if len(selected) >= MIN_SOURCES:
            # Hard floor: below 35% of top → stop
            if score < floor_cutoff:
                break
            # Gap: sharp drop from previous doc → stop
            # Tighten gap after TIGHT_GAP_AFTER docs to prevent long tails
            gap = TIGHT_GAP_RATIO if len(selected) >= TIGHT_GAP_AFTER else GAP_RATIO
            if prev_score > 0 and score < prev_score * gap:
                break

        # Deterministic dedup by content prefix
        content_hash = hashlib.md5(doc["content"][:200].encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        selected.append(doc)
        prev_score = score

    if merged_docs:
        logger.info(
            f"[RELEVANCE] {len(selected)}/{len(merged_docs)} docs selected "
            f"(top={top_score:.4f}, floor={floor_cutoff:.4f}, "
            f"last={selected[-1]['score']:.4f})"
        )

    return selected


def _extract_relevant_excerpt(content: str, question: str, max_chars: int) -> str:
    """Extract the most question-relevant excerpt from a document.

    Rather than truncating from the start, slides a window of
    ``max_chars`` characters over the document and scores each position
    by the total number of Tamil keyword hits (≥ 3 characters, excluding
    common stop words) within the window. Returns the window starting at
    the position with the highest keyword density.

    Falls back to the first ``max_chars`` characters when the document
    is already short enough, when no Tamil keywords can be extracted
    from the question, or when no keyword matches are found anywhere
    in the document.

    Args:
        content (str): Full document text to extract from.
        question (str): User question string; Tamil tokens (≥ 3 chars)
            extracted from it drive the relevance scoring.
        max_chars (int): Size of the excerpt window in characters.

    Returns:
        str: The highest-density excerpt of at most ``max_chars``
        characters, or the full content if it fits within the limit.
    """
    if len(content) <= max_chars:
        return content

    # Extract meaningful Tamil keywords (3+ chars) from the question
    keywords = set(re.findall(r"[\u0B80-\u0BFF]{3,}", question.lower()))
    # Remove common stop-ish words that appear everywhere
    keywords -= {
        "இதழில்",
        "இதழ்",
        "பொன்னி",
        "பொன்னியில்",
        "என்ன",
        "யாவை",
        "எனும்",
        "பற்றி",
        "பற்றிய",
        "என்று",
        "உள்ள",
        "உள்ளது",
    }

    if not keywords:
        return content[:max_chars]

    content_lower = content.lower()

    # Score every position by counting keyword hits in a sliding window
    # Use a coarse step to keep it fast
    step = max(100, max_chars // 10)
    best_pos, best_score = 0, 0

    for pos in range(0, max(1, len(content) - max_chars + 1), step):
        window = content_lower[pos : pos + max_chars]
        score = sum(window.count(kw) for kw in keywords)
        if score > best_score:
            best_score = score
            best_pos = pos

    if best_score == 0:
        return content[:max_chars]

    return content[best_pos : best_pos + max_chars]


def build_context_from_docs(
    relevant_docs: List[Dict],
    question: str = "",
    max_context_chars: int = 15000,
    max_context_docs: int = 15,
) -> Tuple[str, int]:
    """Build the LLM context string from the top relevant documents.

    Caps the document set at ``max_context_docs`` (default 15) so each
    document receives a meaningful character budget (at least 500 chars).
    For each document, extracts the most question-relevant excerpt via
    :func:`_extract_relevant_excerpt` and formats it with a numbered
    header (``"ஆவணம் idx/n — title"``). Prepends a Tamil header line
    instructing the LLM to use all provided documents.

    The per-document character limit is ``max(500, max_context_chars // n)``
    where ``n`` is the number of context documents, distributing the total
    budget evenly.

    Args:
        relevant_docs (List[Dict]): Filtered and scored document dicts,
            already sorted by relevance. Each must contain ``"content"``
            and optionally ``"heading"``.
        question (str): User question used for excerpt extraction.
            Defaults to ``""`` (no keyword-guided windowing).
        max_context_chars (int): Total character budget distributed across
            all context documents. Defaults to ``15000``.
        max_context_docs (int): Maximum number of documents included in
            the context. Defaults to ``15``.

    Returns:
        Tuple[str, int]: A two-element tuple of the assembled context string
        (documents joined by double newlines) and the number of documents
        included. Returns ``("", 0)`` if ``relevant_docs`` is empty.
    """
    if not relevant_docs:
        return "", 0

    # Cap docs sent to LLM — evidence can be larger, context must be focused
    context_docs = relevant_docs[:max_context_docs]
    n = len(context_docs)
    per_doc_limit = max(500, max_context_chars // n)

    context_parts = [f"[{n} ஆவணங்கள் — ஒவ்வொன்றின் தகவலையும் பயன்படுத்தவும்]\n"]
    for idx, doc in enumerate(context_docs, 1):
        excerpt = _extract_relevant_excerpt(doc["content"], question, per_doc_limit)
        title = doc.get("heading", "")
        header = f"ஆவணம் {idx}/{n}"
        if title:
            header += f" — {title}"
        context_parts.append(f"{header}:\n{excerpt}")

    return "\n\n".join(context_parts), n


def format_sources(merged_docs: List[Dict], apply_filter: bool = True) -> List[Dict]:
    """Format merged document dicts into source display records.

    Optionally applies :func:`_select_relevant_docs` score-gap filtering
    before formatting. Passing ``apply_filter=False`` formats the full
    merged pool without filtering — used by the cross-encoder reranker so
    it can evaluate all candidates before performing its own selection.

    Each accepted document is logged at INFO level with its rank, score,
    volume, issue, and heading prefix. The returned dicts contain only the
    fields needed for the evidence panel and downstream reranking.

    Args:
        merged_docs (List[Dict]): Pool of merged document dicts as produced
            by :func:`merge_consecutive_chunks`.
        apply_filter (bool): If ``True``, run :func:`_select_relevant_docs`
            before formatting. If ``False``, format all documents as-is.
            Defaults to ``True``.

    Returns:
        List[Dict]: List of source dicts, each containing: ``"volume"``,
        ``"heading"``, ``"doc_issue"``, ``"author_name"``, ``"content"``,
        ``"word_count"``, ``"chunks_merged"``, ``"score"``, and ``"tags"``.
    """
    relevant = _select_relevant_docs(merged_docs) if apply_filter else merged_docs

    logger = logging.getLogger(__name__)
    sources = []
    for idx, doc in enumerate(relevant):
        logger.info(
            f"[EVIDENCE {idx+1}/{len(relevant)}] "
            f"score={doc['score']:.4f} | "
            f"volume={doc['volume']} issue={doc['doc_issue']} | "
            f"heading={doc['heading'][:60]}"
        )
        sources.append(
            {
                "volume": doc["volume"],
                "heading": doc["heading"],
                "doc_issue": doc["doc_issue"],
                "author_name": doc.get("author_name", ""),
                "content": doc["content"],
                "word_count": doc["word_count"],
                "chunks_merged": doc["chunk_count"],
                "score": doc["score"],
                "tags": doc.get("tags", []),
            }
        )

    return sources


def format_answer_output(answer: str, sources: List[Dict]) -> str:
    """Format a complete answer with source evidence as a Tamil display string.

    Assembles a human-readable output intended for console display or
    plain-text API responses. The answer is prefixed with ``"பதில்:"``
    (Tamil for "Answer"). When sources are present, appends a
    ``"ஆதாரங்கள்"`` (Sources) section listing each source with its
    issue number, volume, heading, word count, relevance score, and
    full content.

    Args:
        answer (str): LLM-generated or extractive answer text.
        sources (List[Dict]): Formatted source dicts as produced by
            :func:`format_sources`, each expected to contain ``"doc_issue"``,
            ``"volume"``, ``"heading"``, ``"word_count"``, ``"score"``,
            and ``"content"`` keys.

    Returns:
        str: Newline-joined display string containing the answer followed
        by the numbered source evidence block. Returns only the answer
        block (with no source section) when ``sources`` is empty.
    """
    lines = []

    lines.append("பதில்:")
    lines.append(answer)
    lines.append("")

    if sources:
        lines.append(f"\nஆதாரங்கள் ({len(sources)} ஆவணங்கள்):")
        lines.append("")

        for idx, source in enumerate(sources, 1):
            lines.append(f"ஆதாரம் {idx}")

            header_parts = [f"இதழ்: {source['doc_issue']}", f"மலர்: {source['volume']}"]
            if source["heading"]:
                header_parts.append(f"தலைப்பு: {source['heading']}")
            lines.append(" • ".join(header_parts))

            lines.append(
                f"சொற்கள்: {source['word_count']} | பொருத்தம்: {source['score']:.3f}"
            )
            lines.append("")
            lines.append(source["content"])
            lines.append("")

    return "\n".join(lines)
