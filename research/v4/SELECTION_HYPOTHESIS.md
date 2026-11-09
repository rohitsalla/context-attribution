# Candidate selection hypothesis while FP32 jobs are queued

As of the November 8 22:45 EST status check, smoke 8553085 has not started. Array 8553116 and CPU review 8553117 are waiting on dependencies. No FP32 budget/selection result is available. H200 normal queue estimates November 9 01:17:30 EST for the smoke; normal A100 and both preemptable dry runs estimate later dates. Scheduling estimates are not promises.

The narrower candidate is that raw-reference audit transfer depends on both retention of influential source content and changes to selection after source deletion. This is a falsifiable empirical question for our pipelines, not a novelty claim or established causal mechanism. Sources absent from the selected factual memory are especially informative: a frozen selector cannot respond to their deletion, while a refitted selector can change which other content appears. Either contract may agree more closely with raw effects; no improvement from freezing is presumed.

For the forthcoming matched-factual 512 comparison, define dynamic effect D = factual score minus the score after deletion and refitted selection, and frozen effect F = the same factual score minus the score after removal of the deleted source's selected chunks. Then D = F + (D - F) exactly. The difference is a contrast between executable selection policies. It is not a unique causal mediation decomposition; token counts and selected content may both change. Magnitudes need not add because the terms can cancel.

The offline reviewer now reports the absolute policy contrast, sign disagreements above a declared 0.001 effect threshold, and raw-influence mass on unrepresented sources. It still reports the original planned budget and fidelity comparisons. Do not use a large policy contrast alone to claim better attribution. Assess whether it changes raw-reference error, ranks and uncertainty, and retain results that contradict the candidate.

CPU job 8553168 completed successfully and audited all six completed BF16 global-extraction traces (see `SELECTION_COVERAGE.md`): reconstruct chunk ownership with the pinned tokenizer, require exact factual memory text and achieved-length matches, then describe raw and compressed influence on sources with no selected tokens. It consumes no GPU. These prior BF16 measurements do not settle FP32 numerical robustness or the frozen/recomputed contrast.

## Prior-art boundary checked November 8, 2026

- [On the impact of retrieved content representations in RAG Pipelines](https://arxiv.org/abs/2605.30790) reports a controlled representation comparison and identifies answer retention as central to accuracy. Generic claims that evidence retention matters have direct precedent. Our proposed target is transfer of a fixed raw-model intervention response, not answer-retention novelty.
- [CUE-R: Beyond the Final Answer in Retrieval-Augmented Generation](https://arxiv.org/abs/2604.05467) studies evidence interventions, utility axes and trace divergence. Generic claims that answer accuracy misses evidence effects or that perturbation-based RAG analysis is new are unavailable.
- [Source Attribution in Retrieval-Augmented Generation](https://arxiv.org/abs/2507.04480) studies document-level Shapley attribution and approximations. Our singleton-deletion response is not Shapley credit, and elementary selection-policy algebra is not a new general attribution estimator.

These boundaries were checked against the primary arXiv abstracts, not claimed as an exhaustive full-text novelty audit. A paper contribution still requires replicated transfer findings, released-method baselines, a frozen final hypothesis and locked confirmation. No further GPU expansion, learning grid or test evaluation is introduced while the current study waits.
