# Stronger extraction at the anchor's factual allowance

Specified after the first two anchored seeds, before this control runs. Adaptive development, not final confirmation. The primary paired-effect comparisons, coefficients and advancement gates remain unchanged.

Use all 96 Hotpot development questions with the completed seed-1729 reference suite. For each question, take the exact factual memory-position count of its literal 32-content-token/source baseline, including frames. Globally rank and pack source sentences with the existing conventional BM25-style extractor, with that count as its maximum. It receives only question and source text, never gold answers/support labels.

Reuse the exact raw answer targets and teacher scores from the anchor. Match the answerer's deterministic math-attention backend and greedy decoding with repetition penalty one. Independently score factual raw context and abort if reference drift exceeds 0.001 mean log probability. Evaluate all singleton and three preselected joint deletions and QA, reporting achieved positions.

The original factual allowance stays fixed during interventions. Global selection can refill that allowance after deletion, whereas source-local memory loses the deleted source's slots. This is a system-level comparison under the same maximum allowance, not an equal-realized-length comparison on every branch and not an isolated causal effect of learning.

This checks whether gains over the weak equal-source literal baseline survive stronger token allocation. A gain only over literal equal quotas is insufficient for a method claim. Report the primary paired objective and both unpaired coefficients, without choosing a favorable seed or switching primary metrics to ranking after inspecting results.

One separately hashed standalone GPU job, 15-minute cap, after the positional control and before remaining benchmark tasks. It leaves active V4 code unchanged. The full stage maximum becomes 5 hours 55 minutes, below six GPU-hours, with one GPU active. Broader dataset/model trials depend on this diagnostic and the already scheduled benchmark results.
