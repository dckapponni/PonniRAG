# PonniRAG Evaluation Module

Measures the quality of LLM-generated answers against human reference answers
for Tamil literary queries drawn from the Ponni magazine archive (1947–1955).

---

## Architecture

```
Evaluation dataset (.json or .csv)
          |
          v
  dataset.py :: load_dataset()
          |
          +--[live=True]---> hybrid_search.ask_question()  (RAG stack)
          |                         |
          |                   llm_answer populated
          |
          v
  metrics.py :: MetricsCalculator.compute_batch()
          |
          |-- intfloat/multilingual-e5-large  (semantic similarity)
          |-- intfloat/multilingual-e5-large  (BERTScore, same model)
          |-- sacrebleu                        (char-level BLEU-1, BLEU-2)
          |-- rouge-score                      (char-level ROUGE-L)
          |
          v
  evaluate.py :: _build_report()  --> EvaluationReport
          |
          v
  stdout table + optional JSON file (--output)
```

---

## Metrics

### Semantic Similarity (weight: 0.50)

Uses `intfloat/multilingual-e5-large` — the same model already deployed
for indexing and search — to encode both the reference answer and the LLM
answer, then computes cosine similarity between the two 1024-dimensional
embeddings.

Both texts are prefixed with `query:` (the E5 STS convention for symmetric
comparison, distinct from the `passage:` prefix used during indexing).

This metric dominates the composite score because Tamil uses frequent
synonyms and morphological inflections — two answers can be semantically
equivalent while sharing few surface tokens.

### BERTScore F1 (weight: 0.25)

Computed via the `bert_score` library using `intfloat/multilingual-e5-large`
as the reference model (no second model download required). BERTScore
calculates pairwise token-level cosine similarities between reference and
candidate, then reports Precision, Recall, and F1.

`rescale_with_baseline=False` is set because the existing English baselines
are not meaningful for Tamil.

### BLEU-1 and BLEU-2 (weight: 0.10 for BLEU-1)

Computed at the **character level** using sacrebleu with `tokenize="char"`.

Tamil is agglutinative: a single orthographic word encodes subject, object,
tense, and aspect. Morphological variants of the same root appear completely
different at the word level but share most of their characters. Character
n-grams measure this surface overlap more faithfully than word n-grams.

Example: the root "வா" (come) appears in "வந்தான்", "வருகிறான்",
"வரவேண்டும்" — all sharing the character "வ" and often "வ ர" but no
complete word tokens.

BLEU-2 is recorded in per-sample output but not included in the composite
score (it is informational only).

### ROUGE-L (weight: 0.15)

Computed via `rouge-score` at the **character level** (each Unicode
code-point treated as a token, joined with spaces). This preserves the Tamil
Unicode block (U+0B80–U+0BFF) and measures the longest common subsequence
of characters between reference and hypothesis.

### Composite Score

```
composite = 0.50 * semantic_similarity
          + 0.25 * bertscore_f1
          + 0.15 * rouge_l
          + 0.10 * bleu_1
```

Interpretation guide:

| Score      | Quality    | Description                                        |
|------------|------------|----------------------------------------------------|
| >= 0.75    | Excellent  | Answer is semantically equivalent to reference     |
| 0.55–0.75  | Good       | Captures main facts; some phrasing differs         |
| 0.35–0.55  | Fair       | Partial answer; key details may be missing         |
| < 0.35     | Poor       | Answer diverges significantly from reference       |

---

## Dataset Format

### JSON (recommended)

Minimal format — used when LLM answers will be generated live:

```json
[
  {
    "id": "ponni_q1",
    "question": "பொன்னி இதழ் எந்த ஆண்டுகளில் வெளியிடப்பட்டது?",
    "human_answer": "பொன்னி இதழ் 1947 முதல் 1955 வரை வெளியிடப்பட்டது."
  }
]
```

Annotated format — used for curated datasets with pre-filled LLM answers:

```json
[
  {
    "id": "ponni_q1",
    "question": "பொன்னி இதழ் எந்த ஆண்டுகளில் வெளியிடப்பட்டது?",
    "human_answer": "பொன்னி இதழ் 1947 முதல் 1955 வரை வெளியிடப்பட்டது.",
    "llm_answer": "பொன்னி 1947 ஆண்டில் தொடங்கி 1955 வரை வெளியானது.",
    "category": "magazine_facts",
    "notes": "Basic factual question about publication years"
  }
]
```

Field reference:

| Field          | Type             | Required | Description                                         |
|----------------|------------------|----------|-----------------------------------------------------|
| `id`           | string           | No       | Sample identifier (auto-generated if absent)        |
| `question`     | string           | Yes      | Tamil or English question                           |
| `human_answer` | string           | Yes      | Human-authored reference answer; sample skipped if empty |
| `llm_answer`   | string or null   | No       | System answer; populated by `--live` if absent      |
| `category`     | string or null   | No       | Free-text tag (e.g. "history", "literature")        |
| `notes`        | string or null   | No       | Reviewer note; not used in scoring                  |

### CSV

Columns: `id`, `question`, `human_answer` (required), plus optional
`llm_answer`, `category`, `notes`. File must be UTF-8 encoded.

```csv
id,question,human_answer,llm_answer,category
q1,பொன்னி எந்த ஆண்டில் தொடங்கியது?,1947 ஆம் ஆண்டில்,,magazine_facts
```

---

## Usage

All commands are run from the `src/` directory.

### Offline evaluation (dataset has pre-filled llm_answers)

```bash
cd src
python -m evaluation.evaluate \
    --dataset evaluation/sample_dataset.json
```

### Live evaluation (calls the running RAG stack for each question)

Requires Qdrant and Ollama to be running.

```bash
cd src
python -m evaluation.evaluate \
    --dataset evaluation/sample_dataset.json \
    --live
```

### Faster run — skip BERTScore

Useful in CI or when `bert-score` is not installed. Composite score is
recomputed without the BERTScore term.

```bash
cd src
python -m evaluation.evaluate \
    --dataset evaluation/sample_dataset.json \
    --no-bertscore
```

### Save enriched dataset and JSON report

```bash
cd src
python -m evaluation.evaluate \
    --dataset evaluation/sample_dataset.json \
    --live \
    --save-answers evaluation/results_with_answers.json \
    --output evaluation/report.json
```

### Programmatic API

```python
from evaluation.evaluate import run_evaluation

report = run_evaluation(
    dataset_path="evaluation/sample_dataset.json",
    live=False,
    use_bertscore=True,
    save_answers_path=None,
    output_path=None,
)

print(f"Composite: {report.avg_composite_score:.4f}")
print(f"Semantic : {report.avg_semantic_similarity:.4f}")

for entry in report.per_sample:
    print(entry["id"], entry["composite_score"])
```

---

## Adding New Evaluation Questions

1. Open (or create) a dataset file, e.g. `src/evaluation/my_dataset.json`.

2. Add a new entry with at minimum `question` and `human_answer`:

   ```json
   {
     "id": "ponni_q6",
     "question": "பொன்னி இதழில் எத்தனை மலர்கள் உள்ளன?",
     "human_answer": "பொன்னி இதழில் 8 மலர்கள் (தொகுதிகள்) உள்ளன.",
     "category": "magazine_facts"
   }
   ```

3. Guidelines for writing good reference answers:
   - Write in complete sentences.
   - Include the key facts that a correct RAG answer must contain.
   - Aim for 1–4 sentences (matching typical Ponni article excerpt length).
   - Use consistent Tamil orthography (NFC-normalised text is preferred).
   - Avoid copy-pasting directly from articles to prevent inflated lexical
     overlap scores.

4. Run with `--live` to generate and store LLM answers, then inspect the
   per-sample report to identify low-scoring items for debugging.

---

## Dependencies

Add to `requirements.txt` if not already present:

```
sentence-transformers    # already required by the RAG pipeline
bert-score               # pip install bert-score
sacrebleu                # pip install sacrebleu
rouge-score              # pip install rouge-score
```

All four are optional at import time — if any package is missing, the
corresponding metric is silently skipped and returns 0. Use `--no-bertscore`
to explicitly skip BERTScore when `bert-score` is not installed.

---

## Running Tests

```bash
# Unit tests only (no external services required)
pytest src/tests/test_evaluation.py -v

# With coverage report
pytest src/tests/test_evaluation.py --cov=evaluation --cov-report=term-missing
```

The test suite mocks all heavy model calls (E5 embeddings, BERTScore, BLEU,
ROUGE) so no model downloads occur. Tests complete in under 5 seconds on a
standard laptop.

---

## Module Reference

| Module              | Key exports                                                  |
|---------------------|--------------------------------------------------------------|
| `evaluation.dataset`  | `EvalSample`, `load_dataset()`, `save_dataset()`           |
| `evaluation.metrics`  | `MetricsCalculator`, `MetricResult`                        |
| `evaluation.evaluate` | `run_evaluation()`, `EvaluationReport`                     |

Top-level re-exports (importable as `from evaluation import ...`):

```python
from evaluation import (
    run_evaluation,
    EvaluationReport,
    load_dataset,
    EvalSample,
    MetricsCalculator,
)
```
