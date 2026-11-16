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

The prespecified BM25 conclusion now has held-out Hotpot confirmation on two readers. The mechanism and learned-selector comparisons are secondary analyses.

## Direction after the replacement and position controls

Lead with the impact of the deletion protocol on audit fidelity at fixed factual context and answers. The four development studies favor frozen deletion over full refitting. The follow-up points toward an effect of making extra text visible, but the preset numerical check failed in two studies and every equal-length text-identity interval includes zero. Keep that follow-up as an ablation, and confirm the deletion-protocol comparison on held-out questions before making a broader claim. See [Replacement and position results](REPLACEMENT_LAYOUT_RESULTS.md).

## Held-out confirmation and learned selector scope

The frozen Hotpot criterion passed for both Qwen and Llama with the tested BM25-style selector. Mean source-effect error fell by about 25% and 21% respectively on the same 400 held-out questions. The separately run extractive RECOMP development comparison has favorable but inconclusive frozen-minus-recomputed estimates on both readers. Lead with the BM25 held-out result and report the uncertainty beyond BM25. See [verified results](HELDOUT_SELECTION_RESULTS.md).


## Fixed-ranking and RECOMP follow-up

Keep the held-out BM25 result as the lead. On Hotpot development data, fixed-ranking repacking increased MAE by 0.090 on Qwen and 0.099 on Llama relative to frozen deletion. The Llama interval excludes zero. Refitting the rankings after repacking changed MAE by +0.017 and −0.003, with both intervals crossing zero. These earlier comparisons make refilling the leading hypothesis. The direct comparison of the two contributions on the 400-question cohort is pending.

RECOMP favored frozen deletion on both readers in the 96-question development study, with both intervals crossing zero. The new runs pair BM25 and RECOMP against each reader's archived raw responses on the same 400 questions. Keep the learned-selector result scoped to development until that review completes. See [the follow-up protocol and decision rule](HELDOUT_FOLLOWUP_PROTOCOL.md).

## Completed shared-cohort follow-up

Lead with the held-out BM25 result and its extension to extractive RECOMP under the same Hotpot evaluation. Frozen deletion reduces RECOMP MAE by 11.3% on Qwen and 7.9% on Llama. Both paired intervals favor it. With BM25, refilling at fixed rankings and the additional ranking refit both increase error. Their direct difference is uncertain on both readers. The two-selector result concerns the tested packing and deletion policies on this Hotpot cohort. Treat the shared-cohort follow-up as secondary evidence and keep causal claims about replacement contents or positions scoped to the earlier ablation. [Verified follow-up](HELDOUT_FOLLOWUP_RESULTS.md)
