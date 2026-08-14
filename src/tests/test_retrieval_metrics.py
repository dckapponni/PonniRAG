"""Unit tests for retrieval (ranking) metrics.

Values are hand-computed so a regression in any formula is caught immediately.
No Qdrant, models, or network — pure functions only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from evaluation.retrieval_metrics import (  # noqa: E402
    average_precision,
    dcg_at_k,
    evaluate_ranking,
    hit_at_k,
    mean_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Ranking of 5 docs; qrels mark d1 (gain 3), d3 (gain 1), d5 (gain 2) relevant.
RANKING = ["d1", "d2", "d3", "d4", "d5"]
QRELS = {"d1": 3, "d3": 1, "d5": 2}  # 3 relevant docs total


def test_precision_at_k():
    """Precision@k divides relevant hits by k, not by results returned."""
    # top-3 = d1,d2,d3 -> 2 relevant / 3
    assert precision_at_k(RANKING, QRELS, 3) == pytest.approx(2 / 3)
    # top-1 = d1 -> 1/1
    assert precision_at_k(RANKING, QRELS, 1) == pytest.approx(1.0)


def test_recall_at_k():
    """Recall@k divides relevant hits by total relevant in the qrels."""
    # top-3 finds d1,d3 -> 2 of 3 relevant
    assert recall_at_k(RANKING, QRELS, 3) == pytest.approx(2 / 3)
    # top-5 finds all 3
    assert recall_at_k(RANKING, QRELS, 5) == pytest.approx(1.0)


def test_hit_at_k():
    """Hit@k is 1 when any relevant doc is in the top k, else 0."""
    assert hit_at_k(RANKING, QRELS, 1) == 1.0
    assert hit_at_k(["d2", "d4"], QRELS, 2) == 0.0


def test_reciprocal_rank():
    """Reciprocal rank is 1/rank of the first relevant doc (0 if none)."""
    # d1 relevant at rank 1
    assert reciprocal_rank(RANKING, QRELS) == pytest.approx(1.0)
    # first relevant (d3) at rank 2
    assert reciprocal_rank(["d2", "d3", "d1"], QRELS) == pytest.approx(0.5)
    # none relevant
    assert reciprocal_rank(["d2", "d4"], QRELS) == 0.0


def test_average_precision():
    """AP averages Precision@k at each relevant rank, normalised by R."""
    # relevant hits at ranks 1 (P=1/1), 3 (P=2/3), 5 (P=3/5); /3 relevant
    expected = (1.0 + (2 / 3) + (3 / 5)) / 3
    assert average_precision(RANKING, QRELS) == pytest.approx(expected)


def test_dcg_and_ndcg_exponential():
    """DCG/nDCG use exponential gain 2^g-1 with a log2(rank+1) discount."""
    # DCG: d1 gain 2^3-1=7 at rank1 /log2(2)=1; d3 gain 2^1-1=1 at rank3 /log2(4)=2;
    #      d5 gain 2^2-1=3 at rank5 /log2(6)
    dcg = 7 / math.log2(2) + 1 / math.log2(4) + 3 / math.log2(6)
    assert dcg_at_k(RANKING, QRELS, 5) == pytest.approx(dcg)
    # ideal order: gains 7,3,1 at ranks 1,2,3
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKING, QRELS, 5) == pytest.approx(dcg / idcg)


def test_ndcg_linear_gain():
    """With exponential=False, nDCG uses raw linear gains."""
    # linear gains: d1=3@1, d3=1@3, d5=2@5
    dcg = 3 / math.log2(2) + 1 / math.log2(4) + 2 / math.log2(6)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKING, QRELS, 5, exponential=False) == pytest.approx(dcg / idcg)


def test_perfect_ranking_ndcg_is_one():
    """A ranking in ideal gain order scores nDCG 1.0."""
    perfect = ["d1", "d5", "d3"]
    assert ndcg_at_k(perfect, QRELS, 3) == pytest.approx(1.0)


def test_no_relevant_docs_returns_zero():
    """Queries with empty qrels score 0 rather than raising."""
    empty = {}
    assert ndcg_at_k(RANKING, empty, 5) == 0.0
    assert recall_at_k(RANKING, empty, 5) == 0.0
    assert average_precision(RANKING, empty) == 0.0


def test_short_ranking_not_padded_into_error():
    """A ranking shorter than k must not raise; precision still divides by k."""
    # ranking shorter than k must not raise; precision divides by k
    assert precision_at_k(["d1"], QRELS, 5) == pytest.approx(1 / 5)


def test_evaluate_ranking_keys_and_mean():
    """evaluate_ranking emits stable keys and mean_metrics averages them."""
    row = evaluate_ranking(RANKING, QRELS, ks=(1, 3))
    assert set(row) >= {"ndcg@1", "ndcg@3", "recall@3", "mrr", "map"}
    mean = mean_metrics([row, row])
    assert mean["mrr"] == pytest.approx(row["mrr"])
