# Real-model pilot, frozen before reading real-model outcomes

Purpose: assess feasibility and estimate direction, without a publication claim.

Primary model: cached Qwen2.5-7B-Instruct in BF16 on an ARC GPU, no fine-tuning. A local Qwen3.5-4B 8-bit MLX run is an operational smoke test and is analyzed separately. The primary pilot uses the existing 20 fictional items, caps 128/256, greedy generation, two unchanged-input repeats and two deletion repeats, 32 masks per local surrogate, L1 penalty 0.01, signed absolute-L1 composition, and absolute-effect ranking. No tuning follows test outcomes in this pilot.

H1: recursive local ranking can miss sources with consequential full-replay effects. Report maximally influential-source discovery at B=1/2/3, rank correlation, and explicit missed-source examples. Effects are resolved only beyond the unchanged-input noise floor. Distinguish original-correct cases and negligible-effect cases. A single miss establishes only a pilot counterexample; prevalence requires a larger and more varied dataset.

H2: local-guided selective replay improves useful discovery at lower total cost. Compare discovery to exact random-subset expectation and cheap lexical/source-ID selection at the same B. Add candidate-scoring time, calls, and tokens to the selected complete-replay costs, with the common original pipeline excluded from all methods. Compare against the six-source reference, and disclose that timing estimates share offline traces. If local scoring exceeds full-replay cost, H2 is unsupported for this implementation even if discovery improves.

Accuracy is strict city-string exact match. Inspect outputs for truncation, extra explanation, or malformed spans before interpreting the scores. Report rather than silently discard failures.

Important dataset limitation discovered during run preparation: the generated fixtures share the same linguistic template; their family labels group entity variants, not truly distinct template families. Thus the original pilot labels do NOT establish template generalization or a valid final family-disjoint benchmark. Treat all 20 as an exploratory pilot, and do not describe the 12 labeled test items as a held-out linguistic-template test. No natural-text items are included. Both caps may be nonbinding on this short context, so actual generated lengths must be checked before attributing differences to compression.

Next decision: if effects are reproducible and H1 counterexamples occur, build genuine template families and audited natural text. If scores already agree with full replay or H2 fails, report that result. Do not expand to unrelated tasks to find a positive result.
