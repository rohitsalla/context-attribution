# Exploratory mechanism probe

Designed after reviewing eight-question Qwen/Llama outcomes, before this probe runs. This is an additional development diagnostic, not a claim of novel discovery or confirmatory test. It cannot establish the origin of model knowledge.

## Witness motivating the probe

In the Llama pilot, the question about the subject of *Katharine the Great* has the correct answer, Katharine Graham, under all five 512-budget memory methods. Deleting source d3 reduces the raw answer's summed log probability by 10.623 nats, but has effect -0.024 under global compression and -0.009 under source-local abstractive compression. Lexical memory retains a 3.912-nat effect. This shows an actual dependence change on the same answer tokens, not simply a missed citation. It does not alone identify why dependence changed: another paragraph discusses Graham, the question contains a known book title, and a compressor may reconstruct a missing relation from those clues or its parameters.

## Hypotheses

Question-conditioned generative compression may infer missing relations and distribute answer-like content into memory, making the original evidence appear dispensable. Alternative explanations include ordinary evidence selection, truncation, duplicate/partial evidence, changed confidence and prompt sensitivity. The mechanism is not assumed established from one case.

## Fixed diagnostic matrix

Use the first eight frozen development questions, Llama-3.1-8B answering, Qwen2.5-1.5B compression, budget 512. For each question compare original text and a deterministic alias stress transformation. On both, run raw context and four compression methods: question-aware global, global with a blank compression question, question-aware source-local, and source-local lexical. The answerer always receives the actual question; only the compression question is blank in the blind condition. This is an explicit prompt ablation, not a perfectly isolated single-token intervention. Record achieved positions; nominal equality does not imply equal information content or length.

Aliasing consistently replaces exact paragraph titles and capitalized gold entity strings everywhere in the question/passages/answer. Overlapping strings use longest-first matching. It does not resolve partial names, pronouns or all aliases. The transformation uses evaluation labels only to construct a synthetic diagnostic; labels are never inputs to compression or source selection. It is not claimed human-audited or naturally distributed. Raw QA performance on the altered version is essential: a drop in raw solvability cannot be attributed specifically to compression.

For each method generate its factual answer and score all singleton deletion effects on the same fixed raw-answer tokens. Store factual and all deletion memory texts for direct inspection. Run matched unconstrained short-answer question-only probes on both models, separately from the evidence-only no-context prompt. These are prompt-sensitive indicators of available knowledge, not proofs of its presence, absence or origin.

Analyze original/alias interactions, answer correctness, achieved length and effect distortion jointly, including cases where raw and compressed systems both answer correctly. Evidence consistent with prior-assisted relation reconstruction would motivate stronger controlled entity substitutions and competing-compressor studies; it is not enough to declare a new causal mechanism. No hidden-idea narrative is accepted merely because an anecdote is compelling.

Execution: one additional GPU job, 25-minute limit, after the already queued follow-ups. Combined follow-up allocation cap becomes 5 hours 55 minutes (under six additional GPU-hours). The existing V2/V3 source directories remain frozen; this standalone module is separately hashed. Timeouts or missing cases stay explicit.
