# Refilling, reranking and extractive RECOMP on Hotpot

All 16 follow-up shards and review job 8749421 completed. Each reader scored the same 400 questions with the archived raw answer targets. The sampled archive rescoring checks matched exactly. Local aggregation using the frozen revision `77e8d44` reproduced every value within 1e-12.

Positive values below mean an increase in source-effect MAE for the operation shown, or a larger reduction for the first selector in the final row. Intervals are paired 95% question bootstrap intervals from the secondary analysis. The two readers share questions.

| Contrast | Qwen | Llama |
|---|---:|---:|
| BM25 refilling at fixed ranking | +0.1253 [+0.0771, +0.1790] | +0.0429 [+0.0197, +0.0696] |
| Additional BM25 ranking refit | +0.0596 [+0.0210, +0.1090] | +0.0411 [+0.0152, +0.0788] |
| Refilling minus ranking refit | +0.0658 [−0.0096, +0.1346] | +0.0018 [−0.0416, +0.0397] |
| RECOMP repacked minus frozen | +0.0739 [+0.0288, +0.1218] | +0.0284 [+0.0049, +0.0544] |
| BM25 gain minus RECOMP gain | +0.1110 [+0.0416, +0.1844] | +0.0556 [+0.0105, +0.1075] |

Both operational changes increase BM25 error on both readers. The direct contrast leaves their relative contributions unresolved. Refilling at fixed ranking includes any displacement of surviving chunks and changes in attended context length. The ranking-refit contrast measures the additional effect when packing uses the refitted scores.

Frozen deletion also lowers RECOMP MAE by 11.3% on Qwen and 7.9% on Llama. RECOMP uses the released Natural Questions sentence scorer with our source-aware packing. Its recomputation changes packing using independent sentence scores. The BM25 gain is larger on both readers in the direct paired comparison. BM25 combines ranking refits and repacking, so this gain difference compares the tested selector and recomputation policies together. Frozen pooled NMAE is 0.907 and 0.851 for BM25 and 0.932 and 0.904 for RECOMP (Qwen and Llama respectively). The zero-effect predictor has NMAE 1.

Lead with the held-out BM25 confirmation and the consistent direction under the tested extractive RECOMP setup on the same Hotpot cohort. Present the two BM25 interventions as separate contributions and leave their relative dominance unresolved. Scope the result to source-effect MAE under these selectors and readers. The RECOMP experiment and BM25 decomposition are secondary analyses on questions already used in the original confirmation. The replacement-evidence and positional study remains exploratory. [Complete paired review](HELDOUT_FOLLOWUP_REVIEW.json)
