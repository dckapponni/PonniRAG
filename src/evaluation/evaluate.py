"""
Main evaluation pipeline for PonniRAG.

Two modes
---------
1. Offline mode  — the dataset JSON already contains `llm_answer` fields.
   Metrics are computed directly without calling the RAG system.

2. Live mode     — the dataset JSON has questions and human_answers only.
   `ask_question()` is called for each question, the answer is stored back
   into the sample, then metrics are computed.

CLI usage
---------
From the project root:

    cd src
    python -m evaluation.evaluate --dataset evaluation/sample_dataset.json

    # Force live evaluation (ignore pre-stored llm_answers):
    python -m evaluation.evaluate --dataset evaluation/sample_dataset.json --live

    # Skip BERTScore (faster, fewer dependencies):
    python -m evaluation.evaluate \\
      --dataset evaluation/sample_dataset.json --no-bertscore

    # Save enriched dataset (with llm_answers filled in):
    python -m evaluation.evaluate \\
        --dataset evaluation/sample_dataset.json \\
        --live \\
        --save-answers evaluation/results_with_answers.json

    # Output results to a JSON file as well:
    python -m evaluation.evaluate \\
        --dataset evaluation/sample_dataset.json \\
        --output evaluation/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# sys.path manipulation — mirrors the project's established pattern
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent       # src/evaluation/
_SRC_DIR = _EVAL_DIR.parent                       # src/
_DB_DIR = _SRC_DIR / "db"                         # src/db/
for _p in [str(_SRC_DIR), str(_DB_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.dataset import EvalSample, load_dataset, save_dataset  # noqa: E402
from evaluation.metrics import MetricResult, MetricsCalculator  # noqa: E402

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Aggregated evaluation results."""

    dataset_path: str
    total_samples: int
    evaluated_samples: int
    skipped_samples: int

    # Macro-averages
    avg_semantic_similarity: float = 0.0
    avg_bertscore_f1: float = 0.0
    avg_bleu_1: float = 0.0
    avg_bleu_2: float = 0.0
    avg_rouge_l: float = 0.0
    avg_composite_score: float = 0.0

    # Per-question breakdown
    per_sample: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "evaluated_samples": self.evaluated_samples,
            "skipped_samples": self.skipped_samples,
            "averages": {
                "semantic_similarity": round(self.avg_semantic_similarity, 4),
                "bertscore_f1": round(self.avg_bertscore_f1, 4),
                "bleu_1": round(self.avg_bleu_1, 4),
                "bleu_2": round(self.avg_bleu_2, 4),
                "rouge_l": round(self.avg_rouge_l, 4),
                "composite_score": round(self.avg_composite_score, 4),
            },
            "per_sample": self.per_sample,
        }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset_path: str | Path,
    live: bool = False,
    use_bertscore: bool = True,
    save_answers_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> EvaluationReport:
    """
    Run the full evaluation pipeline.

    Args:
        dataset_path: Path to the evaluation dataset (.json or .csv).
        live: If True, call ask_question() for every sample regardless of
              whether llm_answer is already present.
        use_bertscore: If False, skip BERTScore computation (faster, fewer
                       dependencies — useful for CI or quick sanity checks).
        save_answers_path: If provided, save the dataset enriched with LLM
                           answers to this JSON path.
        output_path: If provided, write the full EvaluationReport as JSON to
                     this path.

    Returns:
        EvaluationReport with per-sample scores and macro-averages.
    """
    dataset_path = Path(dataset_path)
    logger.info("Loading dataset from %s", dataset_path)
    samples = load_dataset(dataset_path)

    if not samples:
        logger.error("No valid samples found in dataset. Aborting.")
        return EvaluationReport(
            dataset_path=str(dataset_path),
            total_samples=0,
            evaluated_samples=0,
            skipped_samples=0,
        )

    # ------------------------------------------------------------------
    # Step 1: Fill missing llm_answers via live RAG evaluation
    # ------------------------------------------------------------------
    samples = _fill_llm_answers(samples, live=live)

    # ------------------------------------------------------------------
    # Step 2: Filter to samples that are fully ready
    # ------------------------------------------------------------------
    ready = [s for s in samples if s.is_ready()]
    skipped = len(samples) - len(ready)
    if skipped:
        logger.warning(
            "%d sample(s) skipped because llm_answer is still empty. "
            "Run with --live to populate them.",
            skipped,
        )

    # ------------------------------------------------------------------
    # Step 3: Optionally persist enriched dataset
    # ------------------------------------------------------------------
    if save_answers_path:
        save_dataset(samples, save_answers_path)
        logger.info("Saved enriched dataset to %s", save_answers_path)

    # ------------------------------------------------------------------
    # Step 4: Compute metrics
    # ------------------------------------------------------------------
    logger.info(
        "Computing metrics for %d samples (bertscore=%s) ...",
        len(ready),
        use_bertscore,
    )
    calculator = MetricsCalculator(use_bertscore=use_bertscore)

    ids = [s.id for s in ready]
    refs = [s.human_answer for s in ready]
    hyps = [s.llm_answer for s in ready]  # type: ignore[misc]

    metric_results = calculator.compute_batch(ids, refs, hyps)

    # ------------------------------------------------------------------
    # Step 5: Aggregate
    # ------------------------------------------------------------------
    report = _build_report(
        dataset_path=str(dataset_path),
        samples=ready,
        metric_results=metric_results,
        skipped=skipped,
    )

    # ------------------------------------------------------------------
    # Step 6: Print and optionally save
    # ------------------------------------------------------------------
    _print_report(report, samples=ready)

    if output_path:
        _save_report(report, output_path)

    return report


# ---------------------------------------------------------------------------
# Live evaluation helper
# ---------------------------------------------------------------------------


def _fill_llm_answers(
    samples: List[EvalSample], live: bool
) -> List[EvalSample]:
    """
    Populate llm_answer on samples that lack one.

    If live=True, ALL samples are re-evaluated.
    If live=False, only samples with missing llm_answer are evaluated.

    Falls back gracefully if the RAG system is unavailable (e.g., Qdrant is
    not running), leaving llm_answer as None so the sample gets skipped.
    """
    needs_answer = [
        s for s in samples if live or not s.llm_answer
    ]

    if not needs_answer:
        logger.info("All samples already have llm_answers. Skipping live eval.")
        return samples

    logger.info(
        "Running live evaluation for %d sample(s) ...", len(needs_answer)
    )

    try:
        from hybrid_search import ask_question  # noqa: PLC0415

        logger.info("Loaded ask_question from hybrid_search.")
    except ImportError as exc:
        logger.error(
            "Could not import ask_question from hybrid_search: %s\n"
            "Make sure the FastAPI backend dependencies are installed and "
            "src/db/ is on sys.path.",
            exc,
        )
        return samples
    except Exception as exc:
        logger.error(
            "Unexpected error importing hybrid_search: %s", exc
        )
        return samples

    for sample in needs_answer:
        logger.info(
            "  Querying RAG for sample '%s': %s",
            sample.id,
            sample.question[:60],
        )
        t0 = time.time()
        try:
            result = ask_question(sample.question, use_llm=True)
            sample.llm_answer = result.get("answer", "").strip() or None
            elapsed = time.time() - t0
            logger.info(
                "  -> got %d chars in %.1fs",
                len(sample.llm_answer or ""),
                elapsed,
            )
        except Exception as exc:
            logger.error(
                "  RAG query failed for sample '%s': %s", sample.id, exc
            )
            sample.llm_answer = None

    return samples


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _build_report(
    dataset_path: str,
    samples: List[EvalSample],
    metric_results: List[MetricResult],
    skipped: int,
) -> EvaluationReport:
    """Compute macro-averages and build the EvaluationReport."""
    if not metric_results:
        return EvaluationReport(
            dataset_path=dataset_path,
            total_samples=len(samples) + skipped,
            evaluated_samples=0,
            skipped_samples=skipped,
        )

    def avg(attr: str) -> float:
        vals = [getattr(r, attr) for r in metric_results]
        return sum(vals) / len(vals) if vals else 0.0

    sample_map = {s.id: s for s in samples}
    per_sample = []
    for mr in metric_results:
        s = sample_map.get(mr.sample_id)
        entry = mr.to_dict()
        if s:
            entry["question"] = s.question
            entry["category"] = s.category
        per_sample.append(entry)

    return EvaluationReport(
        dataset_path=dataset_path,
        total_samples=len(samples) + skipped,
        evaluated_samples=len(metric_results),
        skipped_samples=skipped,
        avg_semantic_similarity=avg("semantic_similarity"),
        avg_bertscore_f1=avg("bertscore_f1"),
        avg_bleu_1=avg("bleu_1"),
        avg_bleu_2=avg("bleu_2"),
        avg_rouge_l=avg("rouge_l"),
        avg_composite_score=avg("composite_score"),
        per_sample=per_sample,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "id": 12,
    "semantic": 10,
    "bertscore": 10,
    "bleu1": 8,
    "rouge_l": 8,
    "composite": 10,
    "category": 12,
}


def _print_report(report: EvaluationReport, samples: List[EvalSample]) -> None:
    """Print a human-readable table to stdout."""
    sep = "-" * 80
    print()
    print("=" * 80)
    print("  PonniRAG Evaluation Report")
    print("=" * 80)
    print(f"  Dataset : {report.dataset_path}")
    print(f"  Samples : {report.evaluated_samples} evaluated, "
          f"{report.skipped_samples} skipped "
          f"(total {report.total_samples})")
    print()
    print("  Macro-average scores:")
    print(f"    Semantic similarity : {report.avg_semantic_similarity:.4f}")
    print(f"    BERTScore F1        : {report.avg_bertscore_f1:.4f}")
    print(f"    BLEU-1 (char)       : {report.avg_bleu_1:.4f}")
    print(f"    BLEU-2 (char)       : {report.avg_bleu_2:.4f}")
    print(f"    ROUGE-L (char)      : {report.avg_rouge_l:.4f}")
    print(f"    Composite score     : {report.avg_composite_score:.4f}")
    print()

    if not report.per_sample:
        print("  No per-sample breakdown available.")
        print("=" * 80)
        return

    # Header
    print(sep)
    header = (
        f"{'ID':<12} "
        f"{'Semantic':>9} "
        f"{'BERT-F1':>9} "
        f"{'BLEU-1':>7} "
        f"{'ROUGE-L':>8} "
        f"{'Composite':>10}  "
        f"{'Category':<14}"
    )
    print(header)
    print(sep)

    # Rows
    for entry in report.per_sample:
        row = (
            f"{str(entry.get('id', '')):<12} "
            f"{entry.get('semantic_similarity', 0.0):>9.4f} "
            f"{entry.get('bertscore_f1', 0.0):>9.4f} "
            f"{entry.get('bleu_1', 0.0):>7.4f} "
            f"{entry.get('rouge_l', 0.0):>8.4f} "
            f"{entry.get('composite_score', 0.0):>10.4f}  "
            f"{str(entry.get('category') or ''):<14}"
        )
        print(row)

    print(sep)

    # Interpretation guide
    print()
    print("  Score interpretation guide (composite = weighted average):")
    print("    >= 0.75  Excellent  — answer is semantically equivalent")
    print("    0.55–0.75  Good     — captures main facts, some phrasing differs")
    print("    0.35–0.55  Fair     — partial answer, key details may be missing")
    print("    <  0.35  Poor       — answer diverges significantly from reference")
    print()
    print("  Weights: semantic=0.50, BERTScore-F1=0.25, ROUGE-L=0.15, BLEU-1=0.10")
    print("=" * 80)
    print()


def _save_report(report: EvaluationReport, output_path: str | Path) -> None:
    """Write EvaluationReport to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
    logger.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PonniRAG LLM response quality against human reference answers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a pre-filled dataset (llm_answers already present):
  python -m evaluation.evaluate --dataset evaluation/sample_dataset.json

  # Run live evaluation (calls the RAG system for each question):
  python -m evaluation.evaluate --dataset evaluation/sample_dataset.json --live

  # Save enriched answers and JSON report:
  python -m evaluation.evaluate \\
      --dataset evaluation/sample_dataset.json --live \\
      --save-answers evaluation/results.json \\
      --output evaluation/report.json

  # Skip BERTScore for a faster run:
  python -m evaluation.evaluate \\
      --dataset evaluation/sample_dataset.json --no-bertscore
""",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to evaluation dataset (.json or .csv)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Call ask_question() for each sample even if llm_answer is "
            "already present. Requires a running Qdrant + Ollama stack."
        ),
    )
    parser.add_argument(
        "--no-bertscore",
        dest="use_bertscore",
        action="store_false",
        default=True,
        help="Skip BERTScore computation (faster, no bert-score dependency needed).",
    )
    parser.add_argument(
        "--save-answers",
        metavar="PATH",
        default=None,
        help="Save dataset enriched with LLM answers to this JSON file.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write the full evaluation report as JSON to this file.",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        live=args.live,
        use_bertscore=args.use_bertscore,
        save_answers_path=args.save_answers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
