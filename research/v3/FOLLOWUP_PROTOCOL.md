# Follow-up protocol: find the mechanism before scaling

Fixed 8 November 2026 after inspecting completed Qwen/Llama V2 development results and before any V3 outcome. This is adaptive development, not preregistered confirmation. The 400-question locked split remains unrun and is not opened by the follow-up scripts.

## What the completed pilot supports

The replay implementation passes sampled unchanged/cold checks. It does not establish preservation of raw-context effects. On eight development questions per completed answerer, generated source-local summaries are inferior to strong lexical controls in important quality/cost comparisons. The simple proposed representation is therefore not a demonstrated method win. Both learned canaries reduce training loss from identical initial parameters; neither has been evaluated on new questions yet.

The interesting observation is that exact factual answer agreement can coexist with a different highest-influence source under deletion. For Llama at the nominal 512-token budget, the raw and globally summarized systems are both exactly correct on five questions, but their highest-magnitude singleton effects disagree on four of those five. This is a small exploratory witness set, not a prevalence estimate. Lexical selection is stronger and must remain a central comparator. See `PILOT_REVIEW.md` and its JSON for complete per-condition results.

## Candidate idea and competing explanations

An answer-preserving compressor may still alter the response to missing evidence. A useful learned representation should preserve some of this intervention response on new questions, rather than only learning to reproduce training answers. Candidate mechanism: supervise *paired differences* in teacher/student answer scores under matched evidence removal. Source locality supplies a reuse contract, not raw-effect fidelity.

Competing explanations include extra perturbation supervision, loss scaling, training-example memorization, erased question-relevant evidence, underused memory budgets, answer-format artifacts and near-zero reference effects. The following experiments are designed to expose these explanations. They are not a search for a flattering metric.

## Experiment A: broaden essential controls

Use all 96 existing development questions, budgets 256/512, raw context plus per-source lead and lexical memory, on Qwen2.5-7B, Llama-3.1-8B and Mistral-7B-v0.3. Expected 480 records per profile (96 raw + 384 compressed), 1,440 total. These are planned counts, not completed records. Preserve raw/no-context outputs, exact targets, actual budgets, costs and replay receipts. The V2 runner and source files remain frozen. Eight previously inspected dev examples are included and explicitly not treated as untouched confirmation.

## Experiment B: evaluate the existing optimization canaries

Evaluate both eight-example/64-step checkpoints on the same first 32 development questions. The compressor receives no gold answer/support label. Generate factual answers and compare source-deletion effects on the same fixed answer tokens from the raw model. Include all singleton deletions and three question-ID-hashed source pairs. Pair choice is outcome/label blind. Include per-source lexical memory with eight and 32 content positions per source, counting identical source frames. Short paragraphs may use fewer extractive positions; report achieved usage without artificial padding. These runs diagnose generalization; success on eight training cases is not sufficient.

## Experiment C: matched objective comparison

Frozen Qwen2.5-1.5B source features and Qwen2.5-7B answerer; train only the established attention-pooling/projection module, with 32 soft positions/source. Use 128 frozen training questions, 256 factual warm-up steps and 512 additional steps. Shuffle training order deterministically by epoch/seed. Three seeds: 1729, 1730, 1731. Train three objectives for every seed:

1. Factual CE + 0.01 factual score-distillation loss.
2. The same loss + squared teacher/student score error on an intervened context (unpaired distillation control).
3. The same factual loss + squared error in factual-minus-intervened score differences (paired-effect objective).

All conditions run identical teacher computations and factual/intervened student forwards; factual-only discards the perturbation loss. Coefficients and steps are fixed here, not tuned on V3 outcomes. Record identical initial and post-warm-up parameter hashes within a seed. If warm-up hashes differ, investigate rather than call the comparison matched. All teacher-forced scores use target-length-normalized log probabilities. Gold targets are training supervision only. This is pointwise score distillation, not full-distribution KL matching or a certificate of a theoretical KL assumption.

GPU preflight correction: the preliminary canary evaluation inherited a token repetition penalty that treats text prompts and `inputs_embeds` differently. Before any larger objective run, explicitly set `repetition_penalty=1.0` for **all** raw, lexical and soft generations and record it in each manifest. Repeat canary evaluation under that shared policy; retain the preliminary traces as debugging evidence, excluded from efficacy comparisons. This change addresses an execution confound, not a response to comparative outcome selection. The repeated canary has a five-minute cap; the total follow-up allocation limit, including the mechanism probe, is six additional GPU-hours.

Training uses singleton deletions only. Evaluate each model on the same 32 development questions, singleton deletions and three fixed joint deletions/question. The latter tests transfer to untrained intervention combinations. A finite paired-loss reduction without QA/generalization gains is a failure to establish the method. This stage is a small learning-direction experiment, not the final 512-training-question study.

## Analysis and advancement criteria

Report QA EM/F1, common-target mean-log-probability effect MAE, pooled L1-normalized error (avoid means of unstable per-case ratios), a zero-effect predictor, rank recovery, raw-correct/incorrect and both-correct slices, joint-deletion error, achieved positions, exact cold embedding checks, truncation and generation caps. A zero-effect predictor has singleton normalized error exactly one; outperforming it is a necessary sanity check, not sufficient research evidence.

For objective comparisons, average each question's paired differences across available matched seeds, then bootstrap questions. Report all per-seed differences separately. Do not inflate sample size by treating seeds as additional independent questions. These are development intervals following adaptive design, not final confirmatory tests. The soft evaluation timing includes explicit cold diagnostics and is not presented as a production speedup; a separate optimized cost study is required if quality/fidelity gates pass.

Advance only if paired training improves effect preservation over **both** factual-only and unpaired supervision, with QA loss no worse than two F1 points as a development target. Inspect joint-deletion transfer and the cheap lexical frontier. If all soft methods lose badly to position-matched lexical memory, diagnose architecture/learning before adding larger models. If unpaired distillation explains the gain, drop the special paired-objective novelty claim. If only training loss improves, report overfitting and stop efficacy claims.

## Novelty audit additions

- [CoD, 2510.21631](https://arxiv.org/abs/2510.21631) already uses counterfactual explanations to improve few-shot distillation of teacher decision boundaries. Counterfactual supervision or decision-boundary preservation is not a novel general idea here. The specific source-memory intervention objective needs a direct comparison and clear scope.
- [PAIR, 2609.36526](https://arxiv.org/abs/2609.36526) uses matched counterfactual continuations to localize harmful compression events and adapt prompts. Executed-counterfactual compression diagnosis is established.
- [CICL, 2606.08151](https://arxiv.org/abs/2606.08151) ranks decision-relevant context and compresses selected evidence. Decision-aware compression is established.
- [Size-Fidelity Paradox, 2602.09789](https://arxiv.org/abs/2602.09789) studies compressor-scale faithfulness failures. A larger compressor is not presumed to resolve this study's failures.

These additions supplement the V2 audit. Official ContextCite/AttriBoT comparisons, global soft-memory control, a second natural benchmark, independent review, stronger continuation-distribution distillation and locked-test evaluation remain outstanding. None is represented as already executed.

## Execution budget and safety against accidental scaling

Use a separate frozen V3 ARC workspace; import unchanged V2 code. Read the two immutable V2 training checkpoints through a shared results link. Run a five-minute training smoke test after the existing V2 array completes, then the canary evaluation, then follow-up tasks 1–12 with array concurrency one. Cap the 13 main allocations at 25 minutes each, plus the five-minute smoke test: at most 5.5 additional GPU-hours. Missing/mismatched outputs fail visibly; training jobs require fresh directories. CPU aggregation runs after completion and labels incomplete conditions. No paid hosted API, duplicate speculative GPU allocation, test evaluation or concurrent extra GPU is needed.
