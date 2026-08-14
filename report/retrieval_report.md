# PonniRAG Retrieval Ablation

Scored 41 queries (9 skipped — no relevant docs in qrels).

| System | ndcg@1 | recall@1 | precision@1 | hit@1 | ndcg@3 | recall@3 | precision@3 | hit@3 | ndcg@5 | recall@5 | precision@5 | hit@5 | ndcg@10 | recall@10 | precision@10 | hit@10 | mrr | map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 0.3148 | 0.2014 | 0.4146 | 0.4146 | 0.3795 | 0.3380 | 0.2927 | 0.6585 | 0.4282 | 0.4332 | 0.2780 | 0.7561 | 0.5198 | 0.7113 | 0.2366 | 0.8780 | 0.5637 | 0.4142 |
| dense | 0.4727 | 0.2606 | 0.5610 | 0.5610 | 0.5572 | 0.4469 | 0.4553 | 0.7805 | 0.6099 | 0.6274 | 0.4049 | 0.8780 | 0.6674 | 0.8082 | 0.2854 | 0.9512 | 0.6957 | 0.5827 |
| sparse | 0.0871 | 0.0489 | 0.1220 | 0.1220 | 0.1040 | 0.1027 | 0.1138 | 0.2439 | 0.1650 | 0.2274 | 0.1463 | 0.4634 | 0.2104 | 0.3456 | 0.1171 | 0.6341 | 0.2509 | 0.1442 |

## Significance (paired permutation test on ndcg@10)

| Comparison | Δ mean | p-value | sig (p<0.05) |
|---|---|---|---|
| hybrid vs dense | -0.1476 | 0.0001 | yes |
| hybrid vs sparse | +0.3094 | 0.0001 | yes |
