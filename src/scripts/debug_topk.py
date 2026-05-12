"""Dump top-K hybrid search results for debugging ranking."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "db"))

from hybrid_search import (  # noqa: E402
    CSV_PATH,
    SCORE_THRESHOLD,
    HybridQdrantSearch,
    _build_content_vocab,
    _filter_sources_by_relevance,
    _rerank_or_filter,
    _select_relevant_docs,
    correct_query_spelling,
    format_sources,
    get_qdrant_client,
    merge_consecutive_chunks,
)

DEFAULT_QUERY = (
    "பொன்னியில் உள்ள மே தினம் என்ற தலைப்பு எந்த இலக்கிய வடிவத்தை சுட்டுகிறது"
)


def main(limit: int = 30, query: str = DEFAULT_QUERY, needle: str = None):
    """Run the query and print retrieval, merge, select, filter, and rerank stages."""
    client = get_qdrant_client()
    vocab = _build_content_vocab()
    sq = correct_query_spelling(query, str(CSV_PATH), vocab)
    print(f"corrected query: {sq}\n")

    searcher = HybridQdrantSearch(client)
    pts = searcher.search(sq, limit=limit, score_threshold=SCORE_THRESHOLD)
    print(f"raw retrieval: {len(pts)} points")
    for i, p in enumerate(pts[:limit], 1):
        pl = p.payload or {}
        meta = pl.get("metadata", {}) or {}
        heading = meta.get("heading") or pl.get("heading") or ""
        vol = meta.get("volume_id") or meta.get("volume")
        iss = meta.get("issue_number") or meta.get("issue")
        print(
            f"{i:3d}. score={p.score:.4f} vol={vol} iss={iss} heading={heading[:80]!r}"
        )

    print("\n--- after merge ---")
    merged = merge_consecutive_chunks(client, pts)
    for i, d in enumerate(merged[:20], 1):
        h = d.get("heading") or d.get("metadata", {}).get("heading") or ""
        print(f"{i:3d}. score={d.get('score',0):.4f} heading={h[:80]!r}")

    print("\n--- after _select_relevant_docs ---")
    rel = _select_relevant_docs(merged)
    for i, d in enumerate(rel[:20], 1):
        h = d.get("heading") or d.get("metadata", {}).get("heading") or ""
        print(f"{i:3d}. heading={h[:80]!r}")

    print("\n--- format_sources -> _filter_sources_by_relevance (lexical baseline) ---")
    srcs = format_sources(rel)
    filtered = _filter_sources_by_relevance(query, srcs)
    for i, s in enumerate(filtered[:20], 1):
        print(f"{i:3d}. heading={(s.get('heading') or '')[:80]!r}")

    print("\n--- _rerank_or_filter (cross-encoder, prod path: full merged pool) ---")
    reranked = _rerank_or_filter(query, merged)
    for i, s in enumerate(reranked[:20], 1):
        ce = s.get("rerank_score")
        ce_str = f"ce={ce:.4f}" if ce is not None else "ce=NA(fallback)"
        print(f"{i:3d}. {ce_str} heading={(s.get('heading') or '')[:80]!r}")

    if needle:
        print(f"\nmatches {needle!r} in raw retrieval:")
        for i, p in enumerate(pts, 1):
            pl = p.payload or {}
            meta = pl.get("metadata", {}) or {}
            heading = meta.get("heading") or pl.get("heading") or ""
            title = meta.get("title") or ""
            text = pl.get("text") or pl.get("content") or ""
            if needle in heading or needle in text or needle in title:
                print(
                    f"  raw rank {i}: title={title[:60]!r} "
                    f"heading={heading[:60]!r} "
                    f"chunk_id={meta.get('chunk_id')} doc_id={meta.get('doc_id')}"
                )

        print(f"\nmatches {needle!r} in merged pool:")
        for i, d in enumerate(merged, 1):
            h = d.get("heading") or ""
            c = d.get("content") or ""
            if needle in h or needle in c:
                print(
                    f"  merged rank {i}: score={d.get('score'):.4f} "
                    f"heading={h[:60]!r} word_count={d.get('word_count')}"
                )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    q = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY
    nd = sys.argv[3] if len(sys.argv) > 3 else None
    main(n, q, nd)
