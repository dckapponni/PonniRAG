# PonniRAG Retrieval Evaluation & Ablation

Measures **retrieval** quality (did the retriever surface the right articles?),
separate from the **generation** quality already covered by `metrics.py`
(semantic / BLEU / ROUGE). This is the results backbone for the FIRE resource +
demo paper.

## What it answers

How much does Reciprocal Rank Fusion (RRF) of dense E5 + sparse BM25 buy over
either branch alone? Reported with graded nDCG@k, MRR, Recall@k, Precision@k,
Hit@k, and MAP, plus a paired significance test.

## Quick start (TL;DR)

Prerequisites: Qdrant running with the `qdrant_indexer` collection, project
Python deps installed, and `GEMINI_API_KEY` exported. Everything runs from
`src/`.

```bash
cd src

# 0. Metric unit tests — no Qdrant/network needed (fast sanity check).
pytest tests/test_retrieval_metrics.py -v

# 1. Build qrels: pool the 3 systems + Gemini graded judge → qrels + review CSV.
python -m evaluation.build_qrels \
    --dataset evaluation/ground_truth_data.csv \
    --pool-depth 10 \
    --qrels-out evaluation/qrels.json \
    --review-out evaluation/pool_review.csv

# 2. (Recommended) A human edits the `human_gain` column in pool_review.csv,
#    then rebuild human-authoritative qrels:
python -m evaluation.build_qrels \
    --merge-review evaluation/pool_review.csv \
    --qrels-out evaluation/qrels.reviewed.json

# 3. Run the ablation and write the report (table + significance + xlsx).
python -m evaluation.run_retrieval_ablation \
    --dataset evaluation/ground_truth_data.csv \
    --qrels evaluation/qrels.reviewed.json \
    --limit 10 \
    --markdown-out evaluation/retrieval_report.md \
    --xlsx-out evaluation/retrieval_report.xlsx
```

Step 1 costs ~50 Gemini calls (one per query) and needs Qdrant; step 3 needs
Qdrant only. Skip step 2 to run fully automatically on LLM-only qrels (weaker —
see the validity note below).

### Compute metrics programmatically

The metric functions are pure and usable standalone (e.g. against your own
rankings + qrels), no Qdrant required:

```python
from evaluation.retrieval_metrics import evaluate_ranking, mean_metrics

ranking = ["vol1 || 3 || தலைப்பு-A", "vol1 || 3 || தலைப்பு-B"]  # best first
qrels = {"vol1 || 3 || தலைப்பு-A": 3}                            # graded 0-3
row = evaluate_ranking(ranking, qrels, ks=(1, 3, 5, 10))
print(row["ndcg@10"], row["mrr"], row["map"])

# macro-average across many queries
means = mean_metrics([row, ...])
```

## Modules

| File | Role |
|---|---|
| `retrieval_metrics.py` | Pure IR metric functions (unit-tested, no deps). |
| `retrieval_search.py` | Three systems with one signature: `hybrid`, `dense`, `sparse`. |
| `build_qrels.py` | TREC-style pooling + Gemini graded-relevance judge → qrels + review CSV. |
| `run_retrieval_ablation.py` | Runs all systems, computes metrics, significance test, report. |
| `../tests/test_retrieval_metrics.py` | Hand-computed metric correctness tests. |

## Ablation systems

All three hit the same `qdrant_indexer` collection with the same `type=article`
filter and score threshold, differing only in retrieval signal:

- **hybrid** — production `HybridQdrantSearch`: dense + sparse fused with RRF.
- **dense** — E5 `multilingual-e5-large` dense branch only (`using="dense"`).
- **sparse** — `Qdrant/bm25` sparse branch only (`using="sparse"`).

The retrieval unit is an **article**, keyed `"<volume> || <issue> || <title>"`;
chunk-level hits are de-duplicated to article keys before scoring, using the
same key function as qrels construction.

## Annotation protocol (for the paper's reproducibility section)

1. **Pool.** For each of the 50 queries, take the union of the top-`pool_depth`
   (default 10) article keys from all three systems (round-robin merge, capped
   at 40). Documents outside every pool are assumed non-relevant — the standard
   pooling assumption; disclose it.
2. **Judge.** Gemini 2.5 Flash grades every pooled article on a 0–3 scale
   (0 not relevant · 1 marginal · 2 relevant · 3 perfect), judging the whole
   pool per query in one temperature-0 call with the human reference answer as
   context for calibration.
3. **Adjudicate.** A human reviewer corrects the `human_gain` column in
   `pool_review.csv`. `merge_reviewed_qrels` rebuilds qrels using `human_gain`
   where present, else `llm_gain`, making the final qrels human-authoritative.
4. **Report agreement.** Compute Cohen's κ (LLM vs human) on the reviewed
   sample and state it — this is what makes LLM-assisted judging defensible.

> Validity note: LLM-only qrels are a known reviewer concern. Human adjudication
> of the full pool (or at least a substantial stratified sample with reported κ)
> is what turns this from "AI graded itself" into a citable resource.

## Run it

Requires a running Qdrant with the indexed collection and `GEMINI_API_KEY` set.

```bash
cd src

# 1. Build qrels (pool + LLM judge) + human-review CSV
python -m evaluation.build_qrels \
    --dataset evaluation/ground_truth_data.csv \
    --pool-depth 10 \
    --qrels-out evaluation/qrels.json \
    --review-out evaluation/pool_review.csv

# 2. (human edits human_gain in pool_review.csv, then:)
python -m evaluation.build_qrels \
    --merge-review evaluation/pool_review.csv \
    --qrels-out evaluation/qrels.reviewed.json

# 3. Run the ablation
python -m evaluation.run_retrieval_ablation \
    --dataset evaluation/ground_truth_data.csv \
    --qrels evaluation/qrels.reviewed.json \
    --limit 10 \
    --markdown-out evaluation/retrieval_report.md \
    --xlsx-out evaluation/retrieval_report.xlsx
```

## Metrics

- **nDCG@k** — graded, exponential gain `2^g − 1`, `log2(rank+1)` discount.
  Primary metric at k=10.
- **Recall@k / Precision@k / Hit@k** — binary (gain > 0).
- **MRR** — mean reciprocal rank of the first relevant article.
- **MAP** — mean average precision, normalised by total relevant per query.

Queries with no relevant document in the qrels are skipped from scoring (and
counted) rather than deflating every system equally.

## Significance

Two-sided **paired permutation (randomization) test** on nDCG@10, 10 000
sign-flip iterations, fixed seed (reproducible). Reports Δmean and p-value of
hybrid vs each baseline. No SciPy dependency.

## Tests

```bash
pytest src/tests/test_retrieval_metrics.py -v
```
