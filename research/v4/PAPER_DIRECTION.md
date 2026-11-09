# Current paper direction after the complete FP32 frontier

Provisional question: **Which compression and intervention contracts preserve a raw model's source-deletion responses?** Current results favor studying selection stability; they do not establish a successful new paired-training method or a universal compression law.

## Evidence that determines the direction

- All four FP32 studies completed and passed independent validation. Both Hotpot models improve with 128-to-1024 budgets (61% and 56% lower effect MAE). Neither MuSiQue model reliably improves on that contrast. Do not headline a general larger-budget solution.
- At identical factual memory and QA, frozen selection reduces error by 16%, 22%, 48% and 53% across Qwen/Hotpot, Llama/Hotpot, Qwen/MuSiQue and Llama/MuSiQue. The latter three unadjusted exploratory intervals exclude zero; the first remains uncertain. These are different intervention contracts, not a uniquely correct causal credit assignment.
- Frozen MuSiQue fidelity still loses to zero-effect prediction, and raw QA is only 14–20%. Report the absolute failures alongside relative improvements and all budgets.
- Three exact-state anchored seeds show no reliable unique paired-training advantage. Stronger generated summaries replay correctly in sampled checks but transfer raw effects poorly and strongly underfill their cap. These remain controlled negative/limitation results.
- The positional equivalence gate failed, including one selected FP32 CUDA discrepancy above tolerance. No full positional mechanism result is claimed.

Full measurements, intervals and job receipts are in `FP32_COMPLETE_RESULTS.md` and `FP32_FRONTIER_REVIEW.json`. Same-answer/both-correct and oracle scalar-rescaling analyses are post-hoc diagnostics, not substitutes for the primary metric or valid normalized probability calibration.

## Executable next experiment

Run the already specified `SELECTION_MECHANISM_PROTOCOL.md`: one additional fixed-ranking repacking branch on the same four cohorts. Frozen deletion removes selected chunks without replacement; fixed-ranking repacking preserves the original candidate priority/IDF but repacks surviving candidates; refitted selection changes both ranking and packing. Check archived raw/factual scores and exact factual hashes before interpreting the comparisons. These contrasts still do not isolate post-deletion content, length and positions or constitute unique causal mediation.

Smoke 8560784 gates array 8560785, four 15-minute jobs with concurrency one; CPU aggregation 8560786 follows. Including full new limits, the stage upper bound is 20,900 GPU seconds below 21,600. No new learning grid or locked-test run is scheduled.

## What a strong paper still needs

Separate own-pipeline replay, raw-reference transfer and task quality. Specify the executed intervention, source ownership, numerical backend, fixed target and actual achieved memory for every system. Include zero-effect and raw identity checks, selection-policy alternatives, answer-correctness slices and decision/rank metrics without retrospectively replacing the primary error metric.

Use official released implementations/checkpoints (e.g. RECOMP and LLMLingua-2) under achieved-budget curves before claiming broad compressor comparisons. ContextCite/AttriBoT comparisons are required if claiming estimator efficiency, with matched estimands. Representation/answer retention, attribution-compression frontiers, counterfactual distillation and selection effects have substantial precedent; elementary algebra and cache commutation are not novelty claims. See existing related-work audits in `research/v2` and `SELECTION_HYPOTHESIS.md`.

All findings remain adaptive development on public corpora. A frozen final hypothesis, independent review and locked confirmation must follow promising replicated evidence. No ICML readiness, universal fidelity limit or unique knowledge-origin claim is asserted.
