"""Vector search and document processing for Tamil document retrieval.

Handles document chunk retrieval, merging, context building,
relevance filtering, and output formatting.
"""

import hashlib
import logging
import re
from typing import Dict, List, Tuple

from embeddings import COLLECTION_NAME
from qdrant_client import QdrantClient, models
from retry import with_qdrant_retry

logger = logging.getLogger(__name__)


def retrieve_all_chunks_for_document(
    client: QdrantClient,
    doc_id: str,
    doc_issue: str,
    volume: str,
    title: str = None,
) -> List[Dict]:
    """Retrieve all chunks for a specific document.

    When ``title`` is provided, results are further restricted to
    chunks whose ``metadata.title`` matches — the schema's ``doc_id``
    represents a magazine *issue*, so filtering by title isolates a
    single article inside the issue. Pass ``None`` (default) to keep
    legacy issue-level retrieval.
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
    """Merge consecutive chunks from the same article.

    ``doc_id`` in this schema identifies a magazine *issue*; an issue
    contains many articles, each with its own ``metadata.title``. The
    merge key therefore includes title so each article is grouped
    independently — without it, all articles in an issue would
    collapse into a single 50k-word "document" and the original
    article boundaries (and headings) would be lost.

    Chunks with an empty/missing title fall back to issue-level
    grouping (legacy behavior), since there is no article boundary
    to use.
    """
    seen_docs = set()
    merged_docs = []

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

        all_chunks = retrieve_all_chunks_for_document(
            client,
            doc_id=metadata.get("doc_id"),
            doc_issue=metadata.get("doc_issue"),
            volume=metadata.get("volume"),
            title=title or None,
        )

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
    """Extract key facts from documents relevant to question."""
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
    """Select relevant documents using score-gap filtering.

    Three-layer relevance filtering:
    1. **Absolute floor**: score >= 35% of the top document's score
    2. **Gap detection**: stop when a doc scores < 40% of the *previous* doc
       (indicates a sharp relevance drop-off between consecutive results)
    3. **Diminishing returns**: after 20 docs, tighten the gap ratio to 50%
       to prevent long tails of marginally relevant results

    Caps at MAX_SOURCES (100), always returns at least MIN_SOURCES (1).
    De-duplicates by content prefix (using deterministic hash).
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
    """Extract the most question-relevant portion of a document.

    Instead of blindly taking the first N characters, this finds the region
    with the highest density of question keywords and centres the excerpt
    window there.  Falls back to the beginning if no keywords match.
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
    """Build LLM context using excerpts from the top relevant documents.

    The evidence set (for user display) can contain up to 100 docs,
    but the LLM context is capped at max_context_docs (default 15)
    to ensure each document gets enough characters (~1000 each) for
    meaningful analysis. Documents are already sorted by score, so
    the top N are the most relevant.

    Returns:
        Tuple of (context_string, doc_count).
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
    """Format source documents for display.

    When apply_filter is True (default), the score-gap filter
    `_select_relevant_docs` runs first — appropriate when the caller
    wants the legacy lexical-style top sources. When False, the full
    merged pool is formatted as-is — used by the cross-encoder
    reranker so it can see all candidates before its own selection.
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
    """Format complete answer with sources for display."""
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
