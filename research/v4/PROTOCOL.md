# V4: exact-state learning and evidence/influence tradeoffs

Adaptive development, specified after completed V3 outcomes, before V4 outcomes. No locked-test access. This stage does not presume a positive learning or paper result.

## A. Repair the comparison

Use one frozen Qwen answerer/encoder pair, unchanged Hotpot training subset (128 examples), 256 factual warm-up steps, 512 steps per branch and all 96 development questions. Compute features/teacher scores once per suite. Train the warm-up **once** and save its pool parameters, full AdamW state, CPU RNG and CUDA RNG. Restore those exact states before every branch; compare hashes after restoration and abort on mismatch. Force deterministic PyTorch algorithms, CUBLAS workspace configuration and SDPA math attention. Before warm-up, compute same-state gradients twice and require exact hash equality. The specific V3 nondeterministic operation is not presumed identified by this fix.

Branches: factual-only, unpaired perturbation distillation at coefficients 0.1 and 1, paired-effect training at coefficients 0.1 and 1. Keep factual CE + 0.01 factual score-distillation in every branch. Every branch computes both factual and perturbed student forwards. Coefficient 0.1 is the primary stable-regularization comparison; coefficient 1 reproduces the stronger V3 auxiliary-loss scale. Report all branches; select neither a coefficient nor a seed retrospectively as the headline. A pure-soft suite with seed 1729 audits the original failure under repaired controls. It is diagnostic, not a multiple-seed method result.

## B. Stable lexical identity instead of an untrained bottleneck

Three seeds (1729–1731) use the same literal lexical memory with at most 32 content positions/source, plus the same source frames. Add bounded, source-local learned residuals to those **existing token embeddings**, using frozen source features and the established pooling/projection module. A scalar tanh gate initializes at zero. At initialization, the entire prompt embedding stream must exactly equal the literal lexical token stream; test this on actual examples before training. Residual RMS per edited token is bounded by the base embedding RMS. No extra input positions are introduced. This is an ordinary residual/gating construction, not an architectural novelty claim.

Compare the five objectives from the same actual warm-up/optimizer/RNG state within each seed, and the identical-position lexical baseline. Model backbones and literal selection are frozen. Labels are training supervision or evaluation only. Evaluate common-raw-target singleton and unseen joint deletions, QA EM/F1, zero-effect controls and exact cold replay. Report paired question-level differences averaged across seeds; show every seed. Shared-state equality is required for objective-effect interpretation. Math-attention results are a separately specified runtime, not a bitwise reproduction of the old SDPA run.

Advance learning only if paired training beats factual-only and unpaired distillation in effect preservation while losing no more than two QA-F1 points, and is useful relative to literal lexical memory. If this fails, do not present it as a successful method or add a large hyperparameter search. A single repaired pure-soft seed cannot establish generality.

## C. Evidence quality versus executed influence

Across Qwen, Llama and Mistral, use all 96 Hotpot development questions and 96 MuSiQue development questions (32 each 2/3/4 hop; pinned mirror, license/provenance limitations in its README). Four systems: raw context, the existing equal-source lexical baseline, stronger global BM25-style sentence packing at cap 512, and an explicitly gold-informed **full-support oracle**. Global selection recomputes after deletion and has no cache-locality claim. The oracle uses Hotpot supporting sentences or MuSiQue supporting paragraphs, with no length cap; it is an evidence-quality ceiling, never a deployable or equal-budget method. Report its actual positions and support granularity.

For each system generate its factual answer and evaluate all singleton interventions on the exact same raw answer tokens, normalized by target length. Record effect distortion, actual memory size, QA (including MuSiQue aliases), both-correct slices and reference influence outside annotated supports. Non-support annotations do not certify causal irrelevance. Generate one diagnostic deletion answer for the highest-magnitude raw-effect source, explicitly selected for diagnostic severity rather than population inference. This separates discrete-answer changes from confidence-only effects. Do not conflate gold evidence with the model's actual dependence.

## D. Stronger generated-memory control

The completed Qwen self-compression pilot improved on the 1.5B compressor, so use Qwen2.5-7B compression for 32 Hotpot development questions on each answerer, global memory at cap 512. Fix greedy decoding/repetition penalty one, score unchanged target tokens, and retain cold replay validation. Do not let a weak small compressor stand in for generative compression in general. This is still exploratory development; larger matched achieved-length/budget and official released-method comparisons remain required.

## Execution and paper gates

New isolated ARC snapshot; old runtimes/results remain intact. Five-minute anchored/determinism smoke first; after success submit tasks 1–13 with concurrency one and 25-minute limits (5.5 GPU-hours maximum including smoke). CPU aggregation retains incomplete flags. Failed deterministic validation stops the learning suite before expensive training; benchmark/oracle work can be scheduled independently if that failure requires a software fix.

Primary direction is an experimentally grounded quality/influence frontier, with learned residual calibration as a candidate repair only if it passes. A strong paper still needs architecture/metric/teacher-quality ablations, released-method baselines, independent review, a frozen final protocol and locked confirmation. We do not claim a causal mechanism, a universal compression limit, a new general distillation idea, or ICML readiness from these development runs.
