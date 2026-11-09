# Current paper direction, after anchored results

Provisional question: **When can compressed context substitute for the original context in a source-influence audit?** The main evidence should be a validated quality/influence/resource frontier, rather than a claimed new paired-training method.

## What the current evidence supports

- Exact cached replay within a compression pipeline and preservation of the raw pipeline's effects are different properties. The experiment separates their targets and validates each against the appropriate reference.
- Three exact-state anchored seeds show a small primary paired-versus-factual improvement, but no reliable advantage beyond unpaired perturbation supervision. The learned method has not passed its advancement gates.
- Stronger ordinary token allocation answers much better than the anchor under its same factual allowance. It has higher effect error at that allowance, so report the frontier rather than naming a blanket winner.
- On Qwen/Hotpot, the same-runtime 353-to-506-position comparison reduces source-effect error by 30.3%; the mean QA change is small but uncertain. The unchanged-answer subset does not establish an independent budget effect. Replication remains necessary.
- Gold-support oracles improve QA across all three Hotpot model families, but their raw-influence agreement differs by model and subgroup. Support labels are not causal truth; no universal evidence-quality/influence tradeoff is claimed.
- The selected CUDA precision check completed: FP32 greatly reduces error, but one of 30 checks fails the fixed equivalence gate. The full positional run correctly skipped. Pause the positional mechanism claim; see `REMAINING_RESULTS.md`.

## Primary experiments to complete

MuSiQue and all three stronger 7B compressor comparisons have completed (see `MUSIQUE_RESULTS.md` and `REMAINING_RESULTS.md`). The generated summaries underfill their cap and do not recover raw effects. The next executable protocol is `FRONTIER_PROTOCOL.md`: four FP32 budget/selection studies on Qwen/Llama and Hotpot/MuSiQue, with a verified raw passthrough endpoint, fixed targets, actual achieved positions and answer-correctness slices. The Qwen/Hotpot FP32 study is now complete: see `FP32_FIRST_RESULTS.md`. Larger budgets improve fidelity, while the frozen-selection contrast remains uncertain; other model/benchmark replications are still pending.

Before attributing any modest gain to evidence-aware learning, include ordinary unpaired distillation, stronger global extraction, and calibration-only controls. Probability-based influence changes can reflect confidence calibration while greedy decisions stay fixed. That is a different declared answering distribution, not automatically a flaw in a probability estimand. Decision-level deletion responses and rank recovery should complement amplitude metrics; do not change the primary metric retrospectively to rescue a weak method.

Use official released compression implementations/checkpoints where compatible (e.g. [RECOMP](https://arxiv.org/abs/2310.04408), [LLMLingua-2](https://arxiv.org/abs/2403.12968), [official LLMLingua code](https://github.com/microsoft/LLMLingua)) before broad compressor comparisons. ContextCite/AttriBoT are required if the final paper makes estimator-efficiency claims; their estimands must match rather than relabeling a custom baseline. Additional baselines are not claimed implemented here.

## Candidate manuscript structure

1. Define own-pipeline replay fidelity, raw-reference transfer, and task quality independently.
2. Specify source interventions as executable contracts: tokens, source identity, positions, attention routing, model/codec revisions, decoding and numerical backend.
3. Establish a multi-model, two-benchmark frontier with fixed targets, achieved budgets, uncertainty and both-correct/incorrect-answer slices.
4. Test why transfer fails: evidence selection, source allocation, calibration, layout and numerical execution. Keep failures that did not isolate a mechanism.
5. Present anchored learning as a controlled candidate repair/negative ablation unless it beats the strongest alternatives. Report all coefficients/seeds and costs.
6. Freeze the final hypothesis and protocol before locked confirmation. No test evaluation has run.

## Limits of the claim

Simple cache commutation, attention pooling, residual edits, counterfactual distillation and positional bias all have prior art. Current observations are adaptive development on shared public corpora, not a novelty certificate, a universal compression lower bound, or proof of unique knowledge origin. No ICML readiness is asserted. A strong paper depends on replicated results and rigorous alternatives, not the preferred narrative.

## CPU validation while the GPU queue is blocked

Run the single worst selected numerical failure in FP32 on a scheduled CPU compute node, with the same 2D/4D mask checks. This uses no GPU and leaves the full CUDA precision diagnostic unchanged. A CPU pass is a semantic cross-check, not a certificate of GPU numerical behavior or a population result. The separate output/source hashes identify this debugging scope.

## While the FP32 queue is pending

The CPU ownership audit exactly reconstructs all six completed BF16 global-extraction studies. On MuSiQue, sources with no selected factual tokens hold 35–40% of raw influence magnitude, and 12–17% of compressed-pipeline influence magnitude is attached to these sources. This motivates a candidate separation of omitted content and selection-policy changes, rather than another paired-learning grid. These are prior BF16 diagnostics, not new FP32 evidence or proof that freezing improves fidelity. See `SELECTION_COVERAGE.md` and `SELECTION_HYPOTHESIS.md`. The four FP32 studies remain queued; do not claim this candidate is established before their results.

The error-location follow-up (`COVERAGE_ERROR.json`) shows that 68–73% of MuSiQue absolute transfer error is on sources having at least some selected tokens. Source-level absence alone does not locate most error. This observation does not rule out omission within selected sources or identify selection changes as the cause. Keep the FP32 matched-factual policy test primary rather than presenting a one-factor explanation.

## First FP32 result and next gate

On Qwen/Hotpot, 128-to-1024 reduces MAE by 61.3%, and the same-answer diagnostic remains favorable. The secondary 512-to-1024 change reduces MAE by 26.9% with uncertain QA change. Frozen selection has a favorable 16.1% point reduction but its interval includes zero. A cross-fitted oracle one-scalar diagnostic does not erase the large-versus-small budget gap; it is not probability calibration. Prioritize the already queued Llama/MuSiQue replications, then released baselines under achieved resource curves. The candidate distinction between answering sufficiency and audit transfer is not yet a general or novel claim.
