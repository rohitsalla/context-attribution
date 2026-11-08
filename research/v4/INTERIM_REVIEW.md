# Interim: first full evidence study, anchored runs prioritized

Snapshot dated 8 November 2026. The full V4 arrays have **not** finished. The repaired paired-effect and lexical-anchored methods do not yet have completed efficacy comparisons. The first anchored seed is now running, followed by the other two anchored seeds and the repaired pure-soft audit. Pending evidence tasks then resume. This changes scheduling only; no runtime, data or training objective was modified.

The completed 96-question Qwen/Hotpot evidence study gives:

| System | QA F1 (%) | Actual mean positions | Pooled effect error |
|---|---:|---:|---:|
| Raw | 58.93 | 1,465.5 | 0 |
| Equal-source lexical | 47.71 | 508.0 | 1.340 |
| Global lexical | 56.83 | 506.2 | 1.109 |
| Gold-support oracle | 82.00 | 102.6 | 1.334 |

The oracle is annotation-informed and has no length cap; it is a diagnostic ceiling, not a deployable or equal-budget method. Global packing improves on equal source quotas, so the old extraction baseline was weak. Oracle answer quality is much higher, while its raw-influence signature remains different. This motivates studying the relationship between evidence quality and reference-behavior preservation; it does not prove a universal tradeoff or a new causal mechanism.

On the **same 41 questions where raw, global lexical and oracle all answer exactly correctly**, global lexical's pooled effect error is 0.530 and the oracle's is 0.853. Rank misses are 7/41 versus 11/41. The paired mean-MAE difference is 0.140, but its exploratory question-bootstrap interval is [-0.003, 0.252], which includes zero. Do not claim a statistically established oracle-versus-extraction difference from this slice. It is post-outcome selection, not population inference.

About 26.9% of pooled raw absolute influence lies outside annotated supports. Those passages may provide partial/duplicate evidence, layout effects or other behavioral signals; annotations do not certify irrelevance. The support-only oracle has exactly zero effects for excluded passages, giving an elementary error floor when the target is the original full-context deletion response. This is a definition-level consequence, not a new theorem.

Reproduce the paired checks with `review_v4_evidence.py`. Data/trace hashes, cohort IDs and metric definitions are in `INTERIM_EVIDENCE.json`. The source hashes of the completed study remain frozen. Other models and MuSiQue are still needed to test generality. The positional-protocol diagnostic is specified separately, before its outcomes.
