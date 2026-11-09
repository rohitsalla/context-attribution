# Completed FP32 frontier: results replicate by regime, not universally

All four jobs completed with exit zero: Qwen/Hotpot 8553116_0 (38m52s), Llama/Hotpot 8553116_1 (41m53s), Qwen/MuSiQue 8553256_0 (35m45s), Llama/MuSiQue 8553256_1 (38m40s). Aggregation 8553117 completed. Independent `review_frontier.py` validates all completed cohorts, identical target spans/raw references, repeatability, the re-executed raw endpoint and matched 512 factual memories/answers. Hotpot has 96 questions/model; MuSiQue has the pre-outcome, runtime-amended hop-balanced 32/model. No locked examples have run.

## Planned budget and selection contrasts

Percent reductions refer to mean per-question singleton effect MAE. Positive is better; intervals below are paired question-bootstrap differences on the original MAE scale (negative is better). Intervals are adaptive, exploratory and unadjusted for multiple comparisons; two models on shared questions are not independent replications of the data.

| Study | 128-to-1024 error reduction | MAE difference [95% interval] | Frozen-512 error reduction | MAE difference [95% interval] |
|---|---:|---|---:|---|
| hotpot_qwen7 | 61.3% | -0.7698 [-1.0847, -0.4816] | 16.1% | -0.1072 [-0.2198, 0.0049] |
| hotpot_llama8 | 55.6% | -0.3878 [-0.7015, -0.1136] | 22.0% | -0.0969 [-0.1608, -0.0457] |
| musique_qwen7 | -8.2% | 0.0368 [-0.2238, 0.3783] | 48.1% | -0.3198 [-0.7349, -0.0632] |
| musique_llama8 | 9.9% | -0.0312 [-0.2319, 0.1438] | 52.7% | -0.2314 [-0.5657, -0.0065] |

The larger-budget finding replicates across both Hotpot models, but not across benchmarks. MuSiQue has no reliable 128-to-1024 effect improvement: Qwen worsens in the point estimate, Llama improves slightly, and both intervals include zero. This falsifies the simple generalization that more memory alone solves audit transfer for these pipelines.

Frozen selection improves the point estimate on all four studies. Three intervals exclude zero; Qwen/Hotpot remains uncertain. Its factual QA is exactly identical to recomputed selection **by construction**, not because we discovered a deployable model with the same accuracy. The comparison changes the intervention contract and post-deletion content/length. It does not establish uniquely correct attribution or identify IDF refitting, replacement packing, positions or numerical sensitivity as the cause.

## Absolute fidelity and task quality

Pooled NMAE compares common raw-target mean-token-log-probability source effects. Predicting zero effects has NMAE 1; raw passthrough is an independently re-executed implementation endpoint, not an attribution solution.

| Study | Raw QA F1 | Recomputed-512 QA F1 | NMAE 128 / 256 / 512 / 1024 | Frozen-512 NMAE |
|---|---:|---:|---|---:|
| hotpot_qwen7 | 58.93% | 57.87% | 2.106 / 1.762 / 1.115 / 0.814 | 0.935 |
| hotpot_llama8 | 71.00% | 64.57% | 1.718 / 1.355 / 1.084 / 0.763 | 0.845 |
| musique_qwen7 | 19.66% | 18.75% | 1.566 / 1.638 / 2.308 / 1.694 | 1.198 |
| musique_llama8 | 13.97% | 7.72% | 1.909 / 1.694 / 2.658 / 1.720 | 1.257 |

All MuSiQue compressed methods, including frozen selection, remain worse than the zero-effect baseline. Raw MuSiQue QA is low (14–20%), limiting claims about faithful reasoning or knowledge attribution. A relative reduction of 48–53% is not a solved fidelity problem. Nonmonotonic MuSiQue curves also preclude selecting only the favorable budget as the headline.

The refreshed post-hoc amplitude diagnostic includes all completed studies, all methods and the same hash-based cross-fitting. It is an oracle development-label rescaling diagnostic, not probability calibration or a deployable baseline. See `FP32_AMPLITUDE_DIAGNOSTIC.json`; no extra GPU call was made.

## Strongest follow-up

Separate fixed-ranking repacking from full refitting, with the same 512 factual memory, fixed targets, cohorts and FP32 execution contract. This is a more consistent empirical direction than a universal larger-budget claim or another paired-training grid. `SELECTION_MECHANISM_PROTOCOL.md` specifies the new branch and planned paired comparisons before its outcomes.

Stage accounting is 17,000 actual GPU seconds completed. The new five-minute smoke and four 15-minute maximum jobs reserve 3,900 more, totaling 20,900 below the existing 21,600-second ceiling. Smoke **8560784** gates array **8560785** (0 Qwen/Hotpot, 1 Llama/Hotpot, 2 Qwen/MuSiQue, 3 Llama/MuSiQue), concurrency one; CPU review **8560786** follows all outcomes. New runtime source `689b00c` passed two cluster tests and the complete 35-test local suite. Jobs are queued at this receipt, not claimed completed.

Official released compression baselines, a careful prior-art audit, a frozen confirmatory protocol and locked evaluation remain necessary for a paper-level claim. Selection stability and elementary effect decompositions are not claimed inventions.
