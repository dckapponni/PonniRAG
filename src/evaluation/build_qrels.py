r"""Build graded relevance judgments (qrels) via TREC-style pooling + LLM judge.

Retrieval metrics (nDCG/MRR/Recall) need a gold set of relevant documents per
query. Exhaustively judging every article against every query is infeasible, so
we use the standard **pooling** method: run all systems under comparison, take
the union of their top-``pool_depth`` results as the judgment pool, and label
only that pool. Documents outside every system's pool are assumed non-relevant
(the standard, well-understood pooling bias — disclosed in the paper).

Labels are produced by a Gemini judge on a graded 0–3 scale, calibrated per
query by judging the whole pool in one call with the human reference answer as
context. The judge output is written both as machine qrels *and* as a
``pool_review.csv`` for human adjudication — a reviewer overwrites the
``human_gain`` column where the LLM erred, and :func:`merge_reviewed_qrels`
rebuilds the final, human-authoritative qrels. Report the LLM/human agreement
(Cohen's kappa on the reviewed sample) in the paper.

Graded scale
------------
* 0 — not relevant: the article does not help answer the question.
* 1 — marginal: touches the topic but does not contain the answer.
* 2 — relevant: contains part of the answer / supporting evidence.
* 3 — perfect: directly and centrally answers the question.

Usage::

    cd src
    python -m evaluation.build_qrels \\
        --dataset evaluation/ground_truth_data.csv \\
        --pool-depth 10 \\
        --qrels-out evaluation/qrels.json \\
        --review-out evaluation/pool_review.csv

    # After a human edits human_gain in pool_review.csv:
    python -m evaluation.build_qrels --merge-review evaluation/pool_review.csv \\
        --qrels-out evaluation/qrels.reviewed.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_EVAL_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EVAL_DIR.parent
_DB_DIR = _SRC_DIR / "db"
for _p in (str(_SRC_DIR), str(_DB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.retrieval_search import RetrievalSystems  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_SNIPPET_CHARS = 900  # per-article text shown to the judge
_MAX_POOL = 40  # hard cap on pool size per query (keeps the judge prompt sane)


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def _fetch_article_meta_and_text(systems: RetrievalSystems, key: str) -> Dict:
    """Fetch title/author/text for one article key by scrolling its chunks.

    Returns a dict with ``title``, ``author``, ``text`` (joined, truncated).
    The key is ``"volume || issue || title"``; we scroll all matching chunks.
    """
    models = systems._models
    volume, issue, title = (part.strip() for part in key.split("||"))
    must = [
        models.FieldCondition(key="type", match=models.MatchValue(value="article")),
        models.FieldCondition(
            key="metadata.volume", match=models.MatchValue(value=volume)
        ),
        models.FieldCondition(
            key="metadata.doc_issue", match=models.MatchValue(value=issue)
        ),
    ]
    if title:
        must.append(
            models.FieldCondition(
                key="metadata.title", match=models.MatchValue(value=title)
            )
        )

    chunks: List[Dict] = []
    author = ""
    offset = None
    while True:
        points, offset = systems._with_retry(
            systems.client.scroll,
            collection_name=systems._collection,
            scroll_filter=models.Filter(must=must),
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for pt in points:
            payload = pt.payload or {}
            meta = payload.get("metadata", {}) or {}
            author = author or str(meta.get("author_name", "") or "")
            chunks.append(
                {
                    "chunk_id": payload.get("chunk_id", 0),
                    "content": (payload.get("content", "") or "").strip(),
                }
            )
        if offset is None:
            break

    chunks.sort(key=lambda c: c["chunk_id"])
    text = " ".join(c["content"] for c in chunks).strip()
    return {"title": title, "author": author, "text": text[:_SNIPPET_CHARS]}


def build_pool(systems: RetrievalSystems, query: str, pool_depth: int) -> List[str]:
    """Union the top-``pool_depth`` article keys from every ablation system."""
    pool: List[str] = []
    seen = set()
    per_system = {name: fn(query, pool_depth) for name, fn in systems.as_dict().items()}
    # Round-robin merge keeps the pool balanced across systems up to the cap.
    for rank in range(pool_depth):
        for name in per_system:
            keys = per_system[name]
            if rank < len(keys) and keys[rank] not in seen:
                seen.add(keys[rank])
                pool.append(keys[rank])
                if len(pool) >= _MAX_POOL:
                    return pool
    return pool


# ---------------------------------------------------------------------------
# Gemini graded-relevance judge
# ---------------------------------------------------------------------------

_JUDGE_INSTRUCTIONS = """\
You are a Tamil information-retrieval relevance assessor for the Ponni magazine
archive (1947-1955). Given a QUESTION, a human REFERENCE ANSWER, and a numbered
list of candidate ARTICLES, assign each article a graded relevance label:

  0 = not relevant (does not help answer the question)
  1 = marginal (mentions the topic but does not contain the answer)
  2 = relevant (contains part of the answer or supporting evidence)
  3 = perfect (directly and centrally answers the question)

Judge relevance to the QUESTION; use the REFERENCE ANSWER only to understand
what a correct answer looks like. Respond with ONLY a JSON array, one object per
article: [{"index": <int>, "gain": <0-3>, "reason": "<short>"}]. No prose.
"""


def judge_pool(query: str, reference: str, pool: List[Dict]) -> Dict[str, int]:
    """Ask Gemini to grade every article in a query's pool in one call.

    Args:
        query: The user question.
        reference: Human reference answer (context for the judge).
        pool: List of ``{"key", "title", "author", "text"}`` dicts.

    Returns:
        Mapping ``{article_key: gain}`` for the pooled articles. Articles the
        judge omits or that fail to parse default to gain 0.
    """
    from google.genai import types as genai_types  # noqa: PLC0415
    from llm import GEMINI_MODEL, _get_gemini_client  # noqa: PLC0415

    listing = []
    for i, art in enumerate(pool):
        listing.append(
            f"[{i}] தலைப்பு: {art['title'] or '(no title)'} | "
            f"ஆசிரியர்: {art['author'] or '(unknown)'}\n{art['text']}"
        )
    user = (
        f"QUESTION:\n{query}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"ARTICLES:\n" + "\n\n".join(listing)
    )

    client = _get_gemini_client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{_JUDGE_INSTRUCTIONS}\n\n{user}",
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    gains: Dict[str, int] = {art["key"]: 0 for art in pool}
    try:
        parsed = json.loads(resp.text)
        for item in parsed:
            idx = int(item["index"])
            gain = max(0, min(3, int(item["gain"])))
            if 0 <= idx < len(pool):
                gains[pool[idx]["key"]] = gain
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Judge parse failed for query %r: %s", query[:50], exc)
    return gains


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def build_qrels(
    dataset_path: str | Path,
    pool_depth: int,
    qrels_out: str | Path,
    review_out: Optional[str | Path],
) -> Dict[str, Dict[str, int]]:
    """Pool, judge, and persist qrels + a human-review CSV.

    Returns the qrels mapping ``{qid: {doc_key: gain}}``.
    """
    samples = load_dataset(dataset_path)
    systems = RetrievalSystems()

    qrels: Dict[str, Dict[str, int]] = {}
    review_rows: List[Dict] = []

    for n, s in enumerate(samples, start=1):
        logger.info("[%d/%d] q=%s", n, len(samples), s.question[:60])
        pool_keys = build_pool(systems, s.question, pool_depth)
        pool = []
        for key in pool_keys:
            meta = _fetch_article_meta_and_text(systems, key)
            pool.append({"key": key, **meta})

        gains = judge_pool(s.question, s.human_answer, pool)
        # Store only positive-gain docs in qrels; 0s are implicit.
        qrels[s.id] = {k: g for k, g in gains.items() if g > 0}

        for art in pool:
            g = gains.get(art["key"], 0)
            review_rows.append(
                {
                    "qid": s.id,
                    "question": s.question,
                    "doc_key": art["key"],
                    "title": art["title"],
                    "author": art["author"],
                    "snippet": art["text"][:200],
                    "llm_gain": g,
                    "human_gain": "",  # reviewer fills this in
                }
            )

    _write_qrels(qrels, qrels_out)
    if review_out:
        _write_review_csv(review_rows, review_out)
    logger.info(
        "Built qrels for %d queries (%d judged doc-pairs).",
        len(qrels),
        len(review_rows),
    )
    return qrels


def merge_reviewed_qrels(
    review_csv: str | Path, qrels_out: str | Path
) -> Dict[str, Dict[str, int]]:
    """Rebuild qrels from a human-reviewed ``pool_review.csv``.

    Uses ``human_gain`` when the reviewer filled it in, otherwise falls back to
    ``llm_gain``. This makes the final qrels human-authoritative.
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with Path(review_csv).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            qid = row["qid"]
            human = (row.get("human_gain") or "").strip()
            gain = int(human) if human else int(row.get("llm_gain") or 0)
            if gain > 0:
                qrels.setdefault(qid, {})[row["doc_key"]] = gain
    _write_qrels(qrels, qrels_out)
    logger.info("Merged reviewed qrels for %d queries.", len(qrels))
    return qrels


def _write_qrels(qrels: Dict[str, Dict[str, int]], path: str | Path) -> None:
    """Serialise qrels to pretty JSON (UTF-8, Tamil preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(qrels, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote qrels to %s", path)


def _write_review_csv(rows: List[Dict], path: str | Path) -> None:
    """Write the human-adjudication CSV (one row per pooled doc-pair)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qid",
        "question",
        "doc_key",
        "title",
        "author",
        "snippet",
        "llm_gain",
        "human_gain",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote review CSV to %s", path)


def _build_argparser() -> argparse.ArgumentParser:
    """Configure the CLI for qrels construction and review merging."""
    p = argparse.ArgumentParser(description="Build/merge PonniRAG retrieval qrels.")
    p.add_argument("--dataset", metavar="PATH", help="Query dataset (.csv/.json).")
    p.add_argument(
        "--pool-depth", type=int, default=10, help="Top-k per system to pool."
    )
    p.add_argument(
        "--qrels-out",
        metavar="PATH",
        default="evaluation/qrels.json",
        help="Where to write qrels JSON.",
    )
    p.add_argument(
        "--review-out",
        metavar="PATH",
        default="evaluation/pool_review.csv",
        help="Where to write the human-review CSV.",
    )
    p.add_argument(
        "--merge-review",
        metavar="PATH",
        default=None,
        help="Rebuild qrels from a reviewed pool_review.csv instead of judging.",
    )
    return p


def main() -> None:
    """CLI entry point: build qrels, or merge a reviewed CSV into final qrels."""
    args = _build_argparser().parse_args()
    if args.merge_review:
        merge_reviewed_qrels(args.merge_review, args.qrels_out)
        return
    if not args.dataset:
        raise SystemExit("--dataset is required unless --merge-review is used.")
    build_qrels(args.dataset, args.pool_depth, args.qrels_out, args.review_out)


if __name__ == "__main__":
    main()
