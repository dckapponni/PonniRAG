r"""Main evaluation pipeline for PonniRAG.

This module orchestrates end-to-end evaluation of a RAG (Retrieval-Augmented
Generation) system by comparing LLM-generated answers against human reference
answers using a suite of NLP metrics (semantic similarity, BLEU, ROUGE-L).

Evaluation Modes:
    **Offline mode** — the dataset already contains pre-filled ``llm_answer``
    fields. Metrics are computed directly without calling the RAG system.

    **Live mode** — the dataset contains questions and ``human_answer`` fields
    only. :func:`ask_question` is called for each question, the response is
    stored back into the sample, and then metrics are computed.

CLI Usage:
    Run from the project root::

        cd src
        python -m evaluation.evaluate --dataset evaluation/sample_dataset.json

        # Force live evaluation (ignore pre-stored llm_answers):
        python -m evaluation.evaluate --dataset evaluation/sample_dataset.json --live

        # Save enriched dataset (with llm_answers filled in):
        python -m evaluation.evaluate \\
            --dataset evaluation/sample_dataset.json \\
            --live \\
            --save-answers evaluation/results_with_answers.json

        # Output results to a JSON file as well:
        python -m evaluation.evaluate \\
            --dataset evaluation/sample_dataset.json \\
            --output evaluation/report.json

Dependencies:
    - ``evaluation.dataset``: :class:`~evaluation.dataset.EvalSample`,
      :func:`~evaluation.dataset.load_dataset`,
      :func:`~evaluation.dataset.save_dataset`
    - ``evaluation.metrics``: :class:`~evaluation.metrics.MetricResult`,
      :class:`~evaluation.metrics.MetricsCalculator`
    - ``hybrid_search``: ``ask_question`` (imported lazily; required only
      in live mode)
    - ``pandas``: Used for Excel report generation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_EVAL_DIR = Path(__file__).resolve().parent  # src/evaluation/
_SRC_DIR = _EVAL_DIR.parent  # src/
_DB_DIR = _SRC_DIR / "db"  # src/db/
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
    """Aggregated evaluation results for a complete dataset run.

    Stores both macro-averaged metric scores across all evaluated samples
    and a per-sample breakdown list. Produced by :func:`run_evaluation` and
    consumed by :func:`_print_report` and :func:`_save_report_csv`.

    Attributes:
        dataset_path (str): Filesystem path to the source dataset file,
            stored as a string for serialisation compatibility.
        total_samples (int): Total number of samples in the dataset,
            including those skipped due to missing answers.
        evaluated_samples (int): Number of samples that had both
            ``human_answer`` and ``llm_answer`` populated and were
            therefore included in metric computation.
        skipped_samples (int): Number of samples excluded from scoring
            because ``llm_answer`` was absent after the evaluation step.
        avg_semantic_similarity (float): Macro-average semantic similarity
            score across all evaluated samples. Defaults to ``0.0``.
        avg_bleu_1 (float): Macro-average character-level BLEU-1 score.
            Defaults to ``0.0``.
        avg_bleu_2 (float): Macro-average character-level BLEU-2 score.
            Defaults to ``0.0``.
        avg_rouge_l (float): Macro-average character-level ROUGE-L score.
            Defaults to ``0.0``.
        avg_composite_score (float): Macro-average of the weighted composite
            score (semantic=0.70, ROUGE-L=0.20, BLEU-1=0.10). Defaults to
            ``0.0``.
        per_sample (List[Dict]): Ordered list of per-sample result
            dictionaries. Each entry is produced by
            :meth:`~evaluation.metrics.MetricResult.to_dict` and augmented
            with ``"question"`` and ``"category"`` fields from the
            corresponding :class:`~evaluation.dataset.EvalSample`.
            Defaults to an empty list.

    Example::

        report = run_evaluation("data/eval.json")
        print(report.avg_composite_score)
        print(report.to_dict()["averages"])
    """

    dataset_path: str
    total_samples: int
    evaluated_samples: int
    skipped_samples: int

    # Macro-averages
    avg_semantic_similarity: float = 0.0
    avg_bleu_1: float = 0.0
    avg_bleu_2: float = 0.0
    avg_rouge_l: float = 0.0
    avg_composite_score: float = 0.0

    # Per-question breakdown
    per_sample: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise the report to a plain dictionary.

        Converts all fields into a JSON-serialisable ``dict``. Floating-point
        averages are rounded to four decimal places. The ``per_sample`` list
        is included as-is (each element is already a ``dict``).

        Returns:
            dict: A dictionary with the following top-level keys:

            - ``"dataset_path"`` *(str)*: Source dataset file path.
            - ``"total_samples"`` *(int)*: Total sample count.
            - ``"evaluated_samples"`` *(int)*: Samples included in scoring.
            - ``"skipped_samples"`` *(int)*: Samples excluded from scoring.
            - ``"averages"`` *(dict)*: Macro-averaged scores with keys
              ``"semantic_similarity"``, ``"bleu_1"``, ``"bleu_2"``,
              ``"rouge_l"``, ``"composite_score"`` — all rounded to 4 d.p.
            - ``"per_sample"`` *(list)*: Per-sample result entries.

        Example::

            report = run_evaluation("data/eval.json")
            import json
            print(json.dumps(report.to_dict(), indent=2))
        """
        return {
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "evaluated_samples": self.evaluated_samples,
            "skipped_samples": self.skipped_samples,
            "averages": {
                "semantic_similarity": round(self.avg_semantic_similarity, 4),
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
    save_answers_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> EvaluationReport:
    """Run the full PonniRAG evaluation pipeline.

    Orchestrates six sequential steps:

    1. **Load** — parse the dataset from disk via
       :func:`~evaluation.dataset.load_dataset`.
    2. **Fill** — populate missing ``llm_answer`` fields by querying the
       RAG system when ``live=True`` or when answers are absent.
    3. **Filter** — separate ready samples (both answers present) from
       skipped ones.
    4. **Persist** — optionally save the enriched dataset to
       *save_answers_path*.
    5. **Score** — compute metrics for all ready samples via
       :class:`~evaluation.metrics.MetricsCalculator`.
    6. **Report** — aggregate scores, print a formatted table to stdout,
       and optionally write an Excel report to *output_path*.

    Args:
        dataset_path (str | Path): Path to the evaluation dataset file.
            Supported formats: ``.json``, ``.csv``. See
            :func:`~evaluation.dataset.load_dataset` for format details.
        live (bool): When ``True``, call ``ask_question()`` for **every**
            sample, overwriting any pre-stored ``llm_answer`` values. When
            ``False`` (default), only samples without an ``llm_answer`` are
            queried. Requires a running Qdrant + Ollama stack.
        save_answers_path (Optional[str | Path]): If provided, all samples
            (including skipped ones) are serialised with their current
            ``llm_answer`` values to this JSON path after live evaluation.
            Useful for caching answers and avoiding repeated RAG queries.
        output_path (Optional[str | Path]): If provided, the evaluation
            report is saved as an Excel (``.xlsx``) file at this path.
            The ``.xlsx`` extension is appended automatically if absent.

    Returns:
        EvaluationReport: Dataclass containing macro-averaged metric scores
        and per-sample breakdowns. Returns an empty report (all counts zero)
        if no valid samples are found in the dataset.

    Example::

        report = run_evaluation(
            dataset_path="data/eval.json",
            live=True,
            save_answers_path="results/eval_answered.json",
            output_path="results/report",
        )
        print(f"Composite score: {report.avg_composite_score:.4f}")
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
    logger.info("Computing metrics for %d samples ...", len(ready))
    calculator = MetricsCalculator()

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
        _save_report_csv(report, ready, output_path)

    return report


# ---------------------------------------------------------------------------
# Live evaluation helper
# ---------------------------------------------------------------------------


def _fill_llm_answers(samples: List[EvalSample], live: bool) -> List[EvalSample]:
    """Populate ``llm_answer`` fields by querying the RAG system.

    Determines which samples need answers based on the *live* flag, then
    lazily imports ``ask_question`` from ``hybrid_search`` and queries it
    for each qualifying sample. Failed queries (import errors, runtime
    exceptions, or empty responses) leave ``llm_answer`` as ``None`` so the
    sample is gracefully skipped during scoring rather than causing a crash.

    Args:
        samples (List[EvalSample]): All samples loaded from the dataset.
            Modified in-place: the ``llm_answer`` attribute of qualifying
            samples is updated with the RAG system's response.
        live (bool): When ``True``, **all** samples are re-evaluated,
            overwriting any existing ``llm_answer`` values. When ``False``,
            only samples where ``llm_answer`` is ``None`` or empty are
            queried.

    Returns:
        List[EvalSample]: The same list passed in, with ``llm_answer``
        fields updated where queries succeeded. Samples whose queries failed
        retain ``llm_answer=None``.

    Note:
        This function is intended for internal use by :func:`run_evaluation`.
        It performs a lazy import of ``ask_question`` to avoid a hard
        dependency on the RAG stack when operating in offline mode.
    """
    needs_answer = [s for s in samples if live or not s.llm_answer]

    if not needs_answer:
        logger.info("All samples already have llm_answers. Skipping live eval.")
        return samples

    logger.info("Running live evaluation for %d sample(s) ...", len(needs_answer))

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
        logger.error("Unexpected error importing hybrid_search: %s", exc)
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
            logger.error("  RAG query failed for sample '%s': %s", sample.id, exc)
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
    """Compute macro-averages and assemble an :class:`EvaluationReport`.

    Calculates the arithmetic mean of each metric across all
    :class:`~evaluation.metrics.MetricResult` objects, then merges per-sample
    metric dictionaries with their corresponding question text and category
    from the source :class:`~evaluation.dataset.EvalSample`.

    Args:
        dataset_path (str): Stringified path to the source dataset, forwarded
            directly to :class:`EvaluationReport`.
        samples (List[EvalSample]): The ready (fully populated) samples that
            were passed to the metrics calculator.
        metric_results (List[MetricResult]): One result object per evaluated
            sample, in the same order as *samples*. May be empty if no
            samples qualified for scoring.
        skipped (int): Count of samples that were excluded from scoring due
            to a missing ``llm_answer``. Added to ``len(samples)`` to derive
            ``total_samples``.

    Returns:
        EvaluationReport: A fully populated report. If *metric_results* is
        empty, all average scores default to ``0.0`` and ``per_sample`` is
        an empty list.

    Note:
        This function is intended for internal use by :func:`run_evaluation`.
    """
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
    "bleu1": 8,
    "rouge_l": 8,
    "composite": 10,
    "category": 12,
}


def _print_report(report: EvaluationReport, samples: List[EvalSample]) -> None:
    """Print a human-readable evaluation summary table to stdout.

    Renders a fixed-width console report with two sections:

    1. **Header block** — dataset path, sample counts, and macro-averaged
       scores for all five metrics.
    2. **Per-sample table** — one row per evaluated sample showing its ID,
       semantic similarity, BLEU-1, ROUGE-L, composite score, and category.

    A score interpretation guide is appended below the table.

    Args:
        report (EvaluationReport): The aggregated report produced by
            :func:`_build_report`.
        samples (List[EvalSample]): The ready samples corresponding to
            entries in ``report.per_sample``. Not used directly for display
            (data comes from ``report.per_sample``), but kept as a parameter
            for potential future enrichment.

    Returns:
        None

    Note:
        This function is intended for internal use by :func:`run_evaluation`.
        Output goes to ``sys.stdout`` via :func:`print`.
    """
    sep = "-" * 80
    print()
    print("=" * 80)
    print("  PonniRAG Evaluation Report")
    print("=" * 80)
    print(f"  Dataset : {report.dataset_path}")
    print(
        f"  Samples : {report.evaluated_samples} evaluated, "
        f"{report.skipped_samples} skipped "
        f"(total {report.total_samples})"
    )
    print()
    print("  Macro-average scores:")
    print(f"    Semantic similarity : {report.avg_semantic_similarity:.4f}")
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
        f"{'BLEU-1':>7} "
        f"{'ROUGE-L':>8} "
        f"{'Composite':>10}  "
    )
    print(header)
    print(sep)

    # Rows
    for entry in report.per_sample:
        row = (
            f"{str(entry.get('id', '')):<12} "
            f"{entry.get('semantic_similarity', 0.0):>9.4f} "
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
    print("  Weights: semantic=0.70, ROUGE-L=0.20, BLEU-1=0.10")
    print("=" * 80)
    print()


def _save_report_csv(
    report: EvaluationReport, samples: List[EvalSample], output_path: str | Path
) -> None:
    """Save the evaluation report as an Excel (``.xlsx``) workbook.

    Builds a :class:`pandas.DataFrame` with one row per evaluated sample,
    combining question/answer text from *samples* with all numeric metric
    scores from ``report.per_sample``. The resulting workbook is written
    to *output_path* (the extension is forced to ``.xlsx`` regardless of
    what was provided). Missing parent directories are created automatically.

    The output sheet contains the following columns (in order):

    ``S.no``, ``Question``, ``Human answer``, ``LLM answer``,
    ``Semantic similarity``, ``BLEU-1``, ``BLEU-2``, ``ROUGE-L``,
    ``Composite score``.

    Args:
        report (EvaluationReport): The aggregated report whose
            ``per_sample`` list provides metric values and sample IDs.
        samples (List[EvalSample]): Ready samples used to look up question
            and answer text by ID. Samples whose ID does not appear in
            ``report.per_sample`` are not included in the output.
        output_path (str | Path): Destination path for the Excel file.
            The suffix is replaced with ``.xlsx`` automatically. Parent
            directories are created if they do not exist.

    Returns:
        None

    Note:
        This function is intended for internal use by :func:`run_evaluation`.
        Requires ``pandas`` and a compatible Excel writer (e.g. ``openpyxl``).
    """
    output_path = Path(output_path).with_suffix(".xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_map = {s.id: s for s in samples}

    rows = []

    for i, entry in enumerate(report.per_sample, start=1):
        s = sample_map.get(entry["id"])

        rows.append(
            {
                "S.no": i,
                "Question": s.question if s else "",
                "Human answer": s.human_answer if s else "",
                "LLM answer": s.llm_answer if s else "",
                "Semantic similarity": entry.get("semantic_similarity", 0),
                "BLEU-1": entry.get("bleu_1", 0),
                "BLEU-2": entry.get("bleu_2", 0),
                "ROUGE-L": entry.get("rouge_l", 0),
                "Composite score": entry.get("composite_score", 0),
            }
        )

    df = pd.DataFrame(rows)

    df.to_excel(output_path, index=False)

    logger.info("Excel report saved to %s", output_path)


def _build_argparser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser for the evaluation script.

    Defines all command-line arguments accepted by :func:`main`, including
    path arguments for the dataset and output files, a boolean flag for live
    evaluation, and a descriptive epilog with usage examples.

    Returns:
        argparse.ArgumentParser: A fully configured parser ready to call
        ``.parse_args()`` on. The parser uses
        :class:`argparse.RawDescriptionHelpFormatter` so that the multi-line
        epilog examples are preserved as written.

    Defined Arguments:
        ``--dataset PATH`` *(required)*: Path to the evaluation dataset
        (``.json`` or ``.csv``).

        ``--live``: Re-query the RAG system for every sample, ignoring
        pre-stored ``llm_answer`` values.

        ``--save-answers PATH``: Write the enriched dataset (with
        ``llm_answer`` fields) to this JSON file.

        ``--output PATH``: Write the full evaluation report as an Excel
        file to this path.

    Note:
        This function is intended for internal use by :func:`main`.
    """
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
    """Parse CLI arguments and launch the evaluation pipeline.

    Entry point for command-line invocation. Delegates argument parsing to
    :func:`_build_argparser` and forwards all parsed values to
    :func:`run_evaluation`. The resulting :class:`EvaluationReport` is not
    returned (it is printed to stdout and optionally saved to disk by
    :func:`run_evaluation` itself).

    Returns:
        None

    Example::

        # Equivalent to running:
        # python -m evaluation.evaluate --dataset data/eval.json --live
        import sys
        sys.argv = ["evaluate", "--dataset", "data/eval.json", "--live"]
        main()
    """
    parser = _build_argparser()
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        live=args.live,
        save_answers_path=args.save_answers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
