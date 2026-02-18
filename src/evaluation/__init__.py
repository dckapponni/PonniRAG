"""
PonniRAG Evaluation Module
==========================
Evaluates LLM response quality by comparing system-generated answers against
human reference answers for Tamil literary content (Ponni magazine, 1947-1955).

Usage
-----
CLI (recommended):
    cd src && python -m evaluation.evaluate --dataset evaluation/sample_dataset.json

Programmatic:
    from evaluation.evaluate import run_evaluation
    report = run_evaluation("evaluation/sample_dataset.json")

Metrics provided:
    - Semantic similarity  (intfloat/multilingual-e5-large, cosine)
    - BERTScore F1         (via bert_score library, uses multilingual model)
    - BLEU-1 / BLEU-2      (sacrebleu, character-level for Tamil)
    - ROUGE-L              (rouge-score, character n-gram overlap)
    - Composite score      (weighted combination of the above)
"""

from evaluation.evaluate import run_evaluation, EvaluationReport  # noqa: F401
from evaluation.dataset import load_dataset, EvalSample  # noqa: F401
from evaluation.metrics import MetricsCalculator  # noqa: F401

__all__ = [
    "run_evaluation",
    "EvaluationReport",
    "load_dataset",
    "EvalSample",
    "MetricsCalculator",
]
