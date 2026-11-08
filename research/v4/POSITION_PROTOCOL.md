# Additional intervention-protocol diagnostic

Designed after the first 96-question Qwen/Hotpot evidence study, before this probe runs. This is adaptive development. Positional sensitivity of language models is established, including [Lost in the Middle](https://arxiv.org/abs/2307.03172) and [Layer-wise Positional Bias](https://arxiv.org/abs/2601.04098); no new general positional-bias claim is made here.

Physical source deletion changes both source content and the positions of later tokens. The existing experiments measure that declared, executed deletion response correctly. A second intervention can remove the same source tokens while retaining all remaining tokens' original position IDs, including the fixed answer continuation. Both are valid but different intervention definitions. The comparison tests protocol dependence, not which is uniquely true causal credit.

Use all 96 existing Qwen/Hotpot raw outputs and exact fixed target tokens from the completed V4 evidence study. Do not regenerate or choose answers. Recompute compact-position and retained-position deletion effects for every source. Factual custom scoring must agree with the existing scorer to mean-log-probability tolerance 0.001 or the probe aborts.

For two ID-hashed deletions per question, independently cross-check retained-position sparse deletion against a full-length attention-masked construction: deleted source positions are masked as keys at all layers, remaining tokens retain original positions. Record all numerical differences and tolerance failures, rather than assuming GPU kernels give identical floating-point scores for the two shapes. Primary compact versus retained-position sparse scores use the same surviving token stream and sequence shape; only position IDs differ. No cached factual hidden states are reused.

Report protocol differences and influence mass outside annotated supports, together with factual wrapper checks, drift from prior deletion scores and dense/sparse numerical disagreement. If numerical differences are comparable to the effects, qualify the result. Do not reinterpret every outside-support effect as a positional artifact or treat support annotations as causal truth.

One separately hashed standalone module leaves active V4 source unchanged. Queue after the repaired learning array and before the remaining evidence tasks, maintaining one active GPU. Ten-minute maximum extends the current stage's allocation ceiling from 5.5 hours to 5 hours 40 minutes. This control is exploratory and does not unlock test evaluation.
