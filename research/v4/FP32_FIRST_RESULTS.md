# First completed FP32 frontier: Qwen on Hotpot

All 96 questions and 576 method records completed in job 8553116_0 (38m52s). This is one adaptive development model/benchmark, not a multi-model replication or locked-test result. The Llama and both 32-question MuSiQue studies are still pending. `review_frontier.py` independently validates matching target tokens, raw references, repeated-score/passthrough receipts and identical 512 frozen/recomputed factual tokens/answers. All checks pass.

| System | Mean actual positions | QA F1 | Mean effect MAE | Pooled NMAE | Top-source hit |
|---|---:|---:|---:|---:|---:|
| raw | 1465.5 | 58.93% | 0.0000 | 0.000 | 100.0% |
| global_128_recomputed | 122.3 | 43.59% | 1.2553 | 2.106 | 51.0% |
| global_256_recomputed | 250.1 | 48.11% | 1.0506 | 1.762 | 50.0% |
| global_512_recomputed | 506.2 | 57.87% | 0.6645 | 1.115 | 62.5% |
| global_1024_recomputed | 1017.1 | 59.30% | 0.4855 | 0.814 | 76.0% |
| global_512_frozen | 506.2 | 57.87% | 0.5573 | 0.935 | 69.8% |

Raw passthrough is a re-executed identity endpoint, not a scientific attribution solution. Zero-effect prediction has NMAE 1. At 1,024, achieved memory is still 69.4% of raw memory, so this does not demonstrate extreme compression with faithful transfer.

## Planned contrasts

Increasing the cap from 128 to 1,024 reduces mean effect MAE by 61.3%. The paired difference is -0.7698, bootstrap interval [-1.0847, -0.4816]. QA F1 increases by 15.71 points, interval [6.04, 25.40]. On the 50 questions with the same normalized factual answer, MAE falls by 0.4984, interval [-0.9155, -0.1670]. That slice is post-outcome selected, not an independent causal estimate or a population claim.

Freezing selection at 512 reduces MAE from 0.6645 to 0.5573 (16.1%), with identical factual QA by construction. Its paired difference is -0.1072, interval [-0.2198, +0.0049], which includes zero. It improves the point estimate and rank recovery but is not a demonstrated reliable method win. Within the 50 both-exactly-correct questions its interval also includes zero.

Selection-policy sensitivity is nevertheless measurable: mean absolute dynamic-minus-frozen effect is 0.4587. There are 115 sign disagreements among 501 source pairs where both effects exceed 0.001 in magnitude. These selected thresholded counts are diagnostic, not independent samples or proof of error causation. The two contracts have different post-deletion lengths/content; no IDF-versus-refill-versus-layout isolation is claimed. Raw influence on completely unrepresented factual sources is only 5.53% here.

## Secondary and post-hoc controls

The 1,024-minus-512 contrast reduces effect MAE by 0.1790 (26.9%), interval [-0.3420, -0.0264]. QA changes by +1.43 points with interval [-6.37, +9.05], so answer-quality equivalence is not established. This secondary contrast was examined after outcomes, with no multiple-comparison correction.

A post-hoc five-fold question-level oracle rescaling diagnostic fits one nonnegative effect scalar on four development folds and evaluates the fifth. It uses raw-effect labels from other dev questions and is not a deployable calibration method, temperature scaling, or a normalized answering distribution. L1-fit optimality is checked against scalars zero and one. Every method uses the same hash folds; no method/fold selection is made. Cross-fitted NMAEs are:

- global_1024_recomputed: 0.747
- global_128_recomputed: 1.000
- global_256_recomputed: 1.001
- global_512_frozen: 0.850
- global_512_recomputed: 1.010

The cross-fitted 1,024-minus-128 MAE difference is -0.1510, conditional bootstrap interval [-0.2354, -0.0674]. This suggests that a single global amplitude adjustment does not remove the observed budget difference. Bootstrap intervals hold the fitted folds/scales fixed and do not account for the entire fitting procedure. This is weaker than ruling out confidence calibration generally; probability/decision-level and released-method controls remain needed.

## Next direction

Finish the existing Llama and MuSiQue replications before expanding the learning grid. The candidate is that answering quality and raw-counterfactual transfer impose different memory demands, and that selection policy can alter the audit even at identical factual memory. The first clause has a one-model development signal; the universal claim and a reliable frozen-selection benefit are not established. Existing frontier/evidence-retention literature limits novelty claims. A final paper still needs released compression baselines, a frozen confirmatory protocol and locked evaluation. No locked example has run.

Scheduling receipt: after Qwen completed, Llama remained pending with a stale array-task-limit reason. The old array now has only this one nonterminal task. Its throttle was refreshed from one to two to clear that gate; there is still only one schedulable GPU task in it. MuSiQue depends explicitly on that task and both replacement MuSiQue tasks retain concurrency one. No extra GPU work was submitted and effective project concurrency remains one. The Llama task is eligible, but the scheduler estimates a long wait; replacement-job dry runs estimate later starts, so its existing priority was retained.
