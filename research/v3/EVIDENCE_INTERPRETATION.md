# What is working, and what is still a hypothesis

Three V2 model profiles have completed: Qwen-7B, Llama-8B and Mistral-7B, each on the same eight development questions (384 records in total). The Qwen self-compression profile is still running at this update. These samples are too small and adaptively inspected to establish a broad effect.

At a nominal 512-token memory cap, source-local generated summaries score QA F1 37.5/61.1/40.7 across those answerers. Source-local lexical memory scores 72.5/88.6/79.6; raw context scores 55.0/87.5/68.3. Generated summaries often underuse the cap; this is a system comparison under a shared maximum, not an equal-length architecture ablation. The generated source-local audit costs approximately 7.06/6.93/5.92 times raw answer-plus-audit time, versus 0.47/0.53/0.45 for lexical memory. The proposed generated-memory system has not established a useful frontier advantage.

## The harder diagnostic

High lexical QA does not imply preservation of source effects. Pooled normalized effect errors at the 512 cap, computed after normalizing each fixed raw target's log scores by its target-token length, are:

| Answerer | Global generated | Source generated | Source lexical |
|---|---:|---:|---:|
| Qwen | 4.084 | 1.454 | 1.089 |
| Llama | 4.635 | 0.800 | 0.813 |
| Mistral | 2.256 | 1.477 | 1.007 |

A predictor that declares every source effect zero scores exactly 1 on this metric when reference mass is nonzero. Thus even lexical memory's strong QA/ranking does not resolve magnitude preservation in this tiny pilot. This does **not** mean its rankings are worse than random: lexical recovers the raw highest-magnitude source on 5/8, 7/8 and 6/8 questions. Ranking recovery and calibrated effect preservation are different goals. The separate relative-error sanity check and trace hashes are in `PILOT_EFFECT_DIAGNOSTIC.json`; reproduce it with `review_v2_effects.py`.

## A useful lead, without a discovery claim

For the *Katharine the Great* witness, stored intervention receipts confirm that the global compressor still states the book–Graham relationship after the passage explicitly providing it has been removed. The remaining text includes a Graham biography. The missing relation could be inferred from query/context clues or supplied from the compressor's parameters; the current receipt alone does not identify its origin. A source-local generative method also largely erases the raw link passage's effect, while lexical memory retains some dependence.

The candidate idea is to train and evaluate a memory representation's **response to evidence changes**, alongside factual answer quality. The important test is whether paired supervision beats factual-only and unpaired perturbation supervision on new questions, without flattening every source effect toward zero. Joint deletions probe transfer beyond the intervention combinations used in training. The mechanism probe separates question conditioning, entity familiarity and discrete answer changes from confidence-only scaling, with the limitations in `MECHANISM_PROTOCOL.md`.

Counterfactual distillation and decision-aware compression already exist; see CoD, PAIR and CICL in `FOLLOWUP_PROTOCOL.md`. This is a narrowed empirical research question, not a first-of-its-kind claim. If ordinary perturbation distillation explains the improvement, the paper must say so. If the cheap controls dominate all trained variants, architecture or objectives need revision before spending on a full confirmation study.
