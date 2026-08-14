"""Retrieval (ranking) metrics for the PonniRAG ablation study.

These are standard IR effectiveness metrics computed against graded relevance
judgments (qrels). They complement the answer-quality metrics in
``metrics.py`` (semantic similarity / BLEU / ROUGE), which measure *generation*
quality but say nothing about whether the *retriever* surfaced the right
documents in the first place.

Terminology
-----------
* **qrels** — mapping ``{doc_key: gain}`` for a single query, where ``gain`` is
  a non-negative graded relevance label (0 = not relevant, higher = more
  relevant). Document keys absent from the mapping are treated as gain 0.
* **ranking** — an ordered list of ``doc_key`` strings as returned by a
  retrieval system, best first, already de-duplicated.

All functions are pure (no I/O, no models) so they are trivially unit-testable
and deterministic. ``@k`` cut-offs never assume the ranking is at least ``k``
long — a short ranking is scored as-is.

Graded gains use the *exponential* gain function ``2**g - 1`` by default (the
standard nDCG formulation of Burges et al. / Jarvelin & Kekalainen); pass
``exponential=False`` for linear gains.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

Qrels = Mapping[str, float]


# ---------------------------------------------------------------------------
# Set-based metrics (binary relevance: gain > 0)
# ---------------------------------------------------------------------------


def _num_relevant(qrels: Qrels) -> int:
    """Count documents with a positive relevance gain in a qrels mapping."""
    return sum(1 for g in qrels.values() if g > 0)


def precision_at_k(ranking: Sequence[str], qrels: Qrels, k: int) -> float:
    """Precision@k — fraction of the top-``k`` results that are relevant.

    Divides by ``k`` (not by the number of results returned), matching the
    standard TREC definition. A ranking shorter than ``k`` is padded
    implicitly with non-relevant slots.
    """
    if k <= 0:
        return 0.0
    top = ranking[:k]
    hits = sum(1 for d in top if qrels.get(d, 0) > 0)
    return hits / k


def recall_at_k(ranking: Sequence[str], qrels: Qrels, k: int) -> float:
    """Recall@k — fraction of all relevant documents found in the top ``k``.

    Returns 0.0 when the query has no relevant documents (undefined recall),
    so such queries can be filtered by the caller rather than skewing means.
    """
    total = _num_relevant(qrels)
    if total == 0:
        return 0.0
    top = ranking[:k]
    hits = sum(1 for d in top if qrels.get(d, 0) > 0)
    return hits / total


def hit_at_k(ranking: Sequence[str], qrels: Qrels, k: int) -> float:
    """Hit@k (a.k.a. Success@k) — 1.0 if any relevant doc is in the top ``k``."""
    top = ranking[:k]
    return 1.0 if any(qrels.get(d, 0) > 0 for d in top) else 0.0


def reciprocal_rank(ranking: Sequence[str], qrels: Qrels) -> float:
    """Reciprocal rank — ``1/rank`` of the first relevant document (0 if none).

    The mean of this over a query set is MRR.
    """
    for i, d in enumerate(ranking, start=1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def average_precision(ranking: Sequence[str], qrels: Qrels) -> float:
    """Average Precision — mean of Precision@k at each relevant retrieved rank.

    Normalised by the total number of relevant documents in the qrels, so
    relevant documents that never appear in the ranking correctly penalise the
    score (they contribute 0). The mean of this over a query set is MAP.
    """
    total = _num_relevant(qrels)
    if total == 0:
        return 0.0
    hits = 0
    summed = 0.0
    for i, d in enumerate(ranking, start=1):
        if qrels.get(d, 0) > 0:
            hits += 1
            summed += hits / i
    return summed / total


# ---------------------------------------------------------------------------
# Graded metrics (nDCG)
# ---------------------------------------------------------------------------


def _gain(g: float, exponential: bool) -> float:
    """Convert a raw relevance label to a DCG gain value."""
    if g <= 0:
        return 0.0
    return (2.0**g - 1.0) if exponential else float(g)


def dcg_at_k(
    ranking: Sequence[str], qrels: Qrels, k: int, exponential: bool = True
) -> float:
    """Discounted Cumulative Gain over the top ``k`` results.

    Uses the ``log2(rank + 1)`` position discount (rank is 1-indexed).
    """
    total = 0.0
    for i, d in enumerate(ranking[:k], start=1):
        gain = _gain(qrels.get(d, 0), exponential)
        if gain:
            total += gain / math.log2(i + 1)
    return total


def ideal_dcg_at_k(qrels: Qrels, k: int, exponential: bool = True) -> float:
    """DCG of the ideal ranking (relevant docs sorted by descending gain)."""
    gains = sorted((g for g in qrels.values() if g > 0), reverse=True)
    total = 0.0
    for i, g in enumerate(gains[:k], start=1):
        total += _gain(g, exponential) / math.log2(i + 1)
    return total


def ndcg_at_k(
    ranking: Sequence[str], qrels: Qrels, k: int, exponential: bool = True
) -> float:
    """Normalised DCG@k in ``[0, 1]`` (0.0 when the query has no relevant docs)."""
    idcg = ideal_dcg_at_k(qrels, k, exponential)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranking, qrels, k, exponential) / idcg


# ---------------------------------------------------------------------------
# Aggregation over a query set
# ---------------------------------------------------------------------------

# Cut-offs reported in the paper. MRR/MAP are rank-position metrics with no k.
DEFAULT_KS = (1, 3, 5, 10)


def evaluate_ranking(
    ranking: Sequence[str],
    qrels: Qrels,
    ks: Sequence[int] = DEFAULT_KS,
    exponential: bool = True,
) -> Dict[str, float]:
    """Compute all metrics for a single (ranking, qrels) pair.

    Returns a flat dict keyed like ``"ndcg@10"``, ``"recall@5"``, ``"mrr"``,
    ``"map"``. Keys are stable so per-system results can be tabulated directly.
    """
    out: Dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(ranking, qrels, k, exponential)
        out[f"recall@{k}"] = recall_at_k(ranking, qrels, k)
        out[f"precision@{k}"] = precision_at_k(ranking, qrels, k)
        out[f"hit@{k}"] = hit_at_k(ranking, qrels, k)
    out["mrr"] = reciprocal_rank(ranking, qrels)
    out["map"] = average_precision(ranking, qrels)
    return out


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Macro-average a list of per-query metric dicts key-by-key."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: sum(row.get(key, 0.0) for row in per_query) / n for key in keys}
