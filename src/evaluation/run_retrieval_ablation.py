r"""Run the retrieval ablation: hybrid (RRF) vs dense-only vs sparse-only.

Loads the query set and the graded qrels (from :mod:`build_qrels`), runs each
system over every query, computes per-query IR metrics
(:mod:`retrieval_metrics`), macro-averages them, and reports a per-system table
plus a paired significance test of the hybrid system against each baseline.

This is the results section of the FIRE resource/demo paper: it quantifies how
much RRF fusion buys over either single retrieval branch alone.

Significance: a two-sided paired **permutation (randomization) test** on the
primary metric (nDCG@10). No SciPy dependency — the exact/approximate p-value is
estimated by randomly swapping each query's paired scores. Standard practice in
IR evaluation (Smucker et al., 2007).

Usage::

    cd src
    python -m evaluation.run_retrieval_ablation \\
        --dataset evaluation/ground_truth_data.csv \\
        --qrels evaluation/qrels.json \\
        --limit 10 \\
        --markdown-out evaluation/retrieval_report.md \\
        --xlsx-out evaluation/retrieval_report.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_EVAL_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EVAL_DIR.parent
_DB_DIR = _SRC_DIR / "db"
for _p in (str(_SRC_DIR), str(_DB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.retrieval_metrics import (  # noqa: E402
    DEFAULT_KS,
    evaluate_ranking,
    mean_metrics,
)
from evaluation.retrieval_search import RetrievalSystems  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

PRIMARY_METRIC = "ndcg@10"
_N_PERMUTATIONS = 10000


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------


def run_ablation(
    dataset_path: str | Path,
    qrels_path: str | Path,
    limit: int = 10,
    ks: Tuple[int, ...] = DEFAULT_KS,
) -> Dict[str, object]:
    """Run all systems over the query set and compute per-system metrics.

    Only queries that have at least one relevant document in the qrels are
    scored (queries with an empty judgment set are uninformative and would
    deflate every system equally). The count of scored/skipped queries is
    reported.

    Returns a dict with ``means`` (per-system macro-averages), ``per_query``
    (system -> qid -> metrics), ``scored``/``skipped`` counts, and the ordered
    list of scored qids.
    """
    samples = load_dataset(dataset_path)
    with Path(qrels_path).open(encoding="utf-8") as fh:
        qrels_all: Dict[str, Dict[str, float]] = json.load(fh)

    systems = RetrievalSystems().as_dict()

    per_query: Dict[str, Dict[str, Dict[str, float]]] = {n: {} for n in systems}
    scored_qids: List[str] = []
    skipped = 0

    for s in samples:
        qrels = qrels_all.get(s.id, {})
        if not any(g > 0 for g in qrels.values()):
            skipped += 1
            continue
        scored_qids.append(s.id)
        for name, fn in systems.items():
            ranking = fn(s.question, limit)
            per_query[name][s.id] = evaluate_ranking(ranking, qrels, ks=ks)

    means = {
        name: mean_metrics([per_query[name][q] for q in scored_qids])
        for name in systems
    }

    return {
        "means": means,
        "per_query": per_query,
        "scored": len(scored_qids),
        "skipped": skipped,
        "qids": scored_qids,
    }


# ---------------------------------------------------------------------------
# Paired permutation significance test
# ---------------------------------------------------------------------------


def permutation_test(
    scores_a: List[float], scores_b: List[float], n: int = _N_PERMUTATIONS
) -> float:
    """Two-sided paired permutation test p-value on the difference of means.

    For each of ``n`` iterations, each paired difference has its sign randomly
    flipped; the fraction of iterations whose absolute mean difference is >= the
    observed absolute mean difference estimates the p-value. Deterministic (uses
    a fixed-seed NumPy generator) so paper numbers are reproducible.
    """
    import numpy as np  # noqa: PLC0415

    diffs = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if diffs.size == 0:
        return 1.0
    observed = abs(diffs.mean())
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(12345)
    signs = rng.choice([-1.0, 1.0], size=(n, diffs.size))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    # +1 smoothing avoids a p-value of exactly 0.
    return float((np.count_nonzero(perm_means >= observed) + 1) / (n + 1))


def significance_vs_hybrid(
    result: Dict[str, object], metric: str = PRIMARY_METRIC
) -> Dict[str, float]:
    """p-value of hybrid vs each baseline on ``metric`` (paired permutation)."""
    per_query = result["per_query"]  # type: ignore[index]
    qids = result["qids"]  # type: ignore[index]
    hybrid_scores = [per_query["hybrid"][q][metric] for q in qids]
    out: Dict[str, float] = {}
    for name in per_query:
        if name == "hybrid":
            continue
        base_scores = [per_query[name][q][metric] for q in qids]
        out[name] = permutation_test(hybrid_scores, base_scores)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _metric_columns(ks: Tuple[int, ...]) -> List[str]:
    """Ordered metric columns for the report tables."""
    cols: List[str] = []
    for k in ks:
        cols += [f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"hit@{k}"]
    cols += ["mrr", "map"]
    return cols


def render_markdown(
    result: Dict[str, object],
    pvalues: Dict[str, float],
    ks: Tuple[int, ...] = DEFAULT_KS,
) -> str:
    """Render the ablation as a Markdown report (table + significance note)."""
    means: Dict[str, Dict[str, float]] = result["means"]  # type: ignore[assignment]
    cols = _metric_columns(ks)
    order = ["hybrid", "dense", "sparse"]

    lines = ["# PonniRAG Retrieval Ablation", ""]
    lines.append(
        f"Scored {result['scored']} queries "
        f"({result['skipped']} skipped — no relevant docs in qrels)."
    )
    lines.append("")
    lines.append("| System | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for name in order:
        if name not in means:
            continue
        row = [f"{means[name][c]:.4f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"## Significance (paired permutation test on {PRIMARY_METRIC})")
    lines.append("")
    lines.append("| Comparison | Δ mean | p-value | sig (p<0.05) |")
    lines.append("|---|---|---|---|")
    hyb = means["hybrid"][PRIMARY_METRIC]
    for name, p in pvalues.items():
        delta = hyb - means[name][PRIMARY_METRIC]
        sig = "yes" if p < 0.05 else "no"
        lines.append(f"| hybrid vs {name} | {delta:+.4f} | {p:.4f} | {sig} |")
    lines.append("")
    return "\n".join(lines)


def save_xlsx(
    result: Dict[str, object], path: str | Path, ks: Tuple[int, ...] = DEFAULT_KS
) -> None:
    """Write a two-sheet workbook: macro-average summary + per-query long form."""
    import pandas as pd  # noqa: PLC0415

    path = Path(path).with_suffix(".xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _metric_columns(ks)
    means: Dict[str, Dict[str, float]] = result["means"]  # type: ignore[assignment]

    summary = pd.DataFrame(
        [{"system": n, **{c: means[n][c] for c in cols}} for n in means]
    )

    per_query = result["per_query"]  # type: ignore[index]
    rows = []
    for name in per_query:
        for qid, m in per_query[name].items():
            rows.append({"system": name, "qid": qid, **m})
    detail = pd.DataFrame(rows)

    with pd.ExcelWriter(path) as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        detail.to_excel(xl, sheet_name="per_query", index=False)
    logger.info("Wrote xlsx report to %s", path)


def _build_argparser() -> argparse.ArgumentParser:
    """Configure the ablation CLI."""
    p = argparse.ArgumentParser(description="Run the PonniRAG retrieval ablation.")
    p.add_argument("--dataset", required=True, metavar="PATH")
    p.add_argument("--qrels", required=True, metavar="PATH")
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Results retrieved per system per query (>= max k).",
    )
    p.add_argument("--markdown-out", metavar="PATH", default=None)
    p.add_argument("--xlsx-out", metavar="PATH", default=None)
    return p


def main() -> None:
    """CLI entry point: run ablation, print + optionally save the report."""
    args = _build_argparser().parse_args()
    result = run_ablation(args.dataset, args.qrels, limit=args.limit)
    pvalues = significance_vs_hybrid(result)
    report = render_markdown(result, pvalues)
    print("\n" + report + "\n")
    if args.markdown_out:
        Path(args.markdown_out).write_text(report, encoding="utf-8")
        logger.info("Wrote markdown report to %s", args.markdown_out)
    if args.xlsx_out:
        save_xlsx(result, args.xlsx_out)


if __name__ == "__main__":
    main()
