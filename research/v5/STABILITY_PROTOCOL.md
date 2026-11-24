# Prospective effect-rescaling, coverage and replacement study

Frozen before any new reader scores. Prior scientific reference: completed generalization study b11f39e. This study is a new hypothesis test, not a continuation of held-out claims using adapted settings.

## Questions and hypotheses

H1: At the primary 512-token budget, frozen deletion has lower raw-deletion attribution error than recomputing selection even after both methods receive independently fitted nonnegative effect scales from separate development questions. This tests robustness to an amplitude/shrinkage explanation. A failure to establish the contrast does not prove the entire prior gain was caused by shrinkage.

H2: Rescaled frozen deletion beats predicting zero effects in absolute attribution error. A relative advantage over recompression alone is insufficient for an absolute usefulness claim. Zero predictions are an error sanity check and cannot meaningfully rank sources.

Secondary: characterize how the contrasts, omitted-evidence error and ranking change over budgets. Test replacement-content sensitivity at matched branch length using declared cross-question donors. These diagnostics do not identify a unique natural causal effect or pure length effect.

## Cohorts and reader matrix

Use Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 at revisions pinned in research/v4/GENERALIZATION_MODELS.json. This is a two-family mechanism study, not a claim over a population of models. All weights are already cached on ARC; no paid APIs or new model weights are needed.

Prepare 324 new MuSiQue questions (84 dev, 240 test; equal allocation by 2/3/4 hops) and 320 new 2Wiki questions (80 dev, 240 test; equal allocation by four types). Exclude normalized questions from every JSONL under the existing project data directory. Select by fixed ID hash within strata. Donor questions use a separate fixed hash and come from the same benchmark and split. Publish counts, source revisions and output hashes. Mirrors are revision-pinned; official byte equivalence, pretraining overlap and document contamination are not verified. Reuse the licenses and mirror caveats of the preceding study.

Each reader evaluates the same 644 questions at budgets 256, 512, 1024 and 2048: 1,288 distinct reader-question evaluations, or 5,152 reader-question-budget configurations. Repeated budgets and readers are not independent question identities. Eight stride shards per benchmark contain both its dev and test questions, yielding 32 full GPU jobs.

## Computation and contracts

Use the preceding Python 3.10 / PyTorch 2.10.0+cu128 / Transformers 5.17.0 runtime (STABILITY_RUNTIME.json), preserving the virtual-environment interpreter path. Validate it before submission and inside allocations. Use the preceding FP32 reader and RECOMP encoder, forced SDPA math, TF32 disabled, deterministic algorithms, explicit 4D causal masks, compact positions and double log-softmax. Generate one raw greedy factual answer per question and use the identical target tokens for every policy, omission and budget. Score every singleton source deletion plus two source pairs selected by the preceding fixed hash. Repeated raw factual and one re-executed raw-deletion score must match within 1e-7. Empty raw targets or nonfinite scores fail explicitly. QA reports use each selector's greedy factual answer separately; they are not the attribution target.

Policies: BM25 frozen deletion, fixed-ranking repacking, refitted ranking/repacking; RECOMP frozen deletion and repacking of independently scored sentences. Record raw effects, all compressed effects, token lengths, retained tokens per source, newly admitted tokens and displaced surviving tokens. Source ownership remains explicit. Sources with zero selected tokens must have exactly zero frozen effects.

At 512 tokens, for each selector choose one retained source by fixed hash without reading its effect. Build the fixed-ranking refill branch. Replace only its newly admitted chunks with a deterministic stream from the donor question, preserving chunk token counts, shared surviving chunks, total length and compact positions. Store actual and donor scores. Donor text is not neutral padding and can accidentally be relevant. The actual-minus-donor contrast tests replacement content at fixed length; frozen-versus-donor mixes content and length and cannot be called a pure length effect. Refilling can also displace surviving chunks, which is recorded. The main study does not rely on the previously inconclusive sparse/dense positional controls.

## Development-only effect rescaling

For each fixed reader, benchmark, budget and policy independently, fit alpha >= 0 on development rows to minimize mean per-question singleton L1 effect error. Use the weighted median of raw_effect / compressed_effect with weights abs(compressed_effect) / number_of_sources_in_question, clipped below at zero; all-zero predictors use alpha=0. Verify fitted loss is no worse than alpha=0 or alpha=1. This is effect rescaling using development raw-effect labels, not probability or temperature calibration. No test effects, QA labels or test-selected parameters enter the fit. Source pair effects are saved as secondary data and do not fit alpha.

Evaluate unscaled and rescaled question MAE, pooled NMAE against the zero predictor, and top-source recovery. Top-source comparisons with tied predictions use fractional tie credit; exclude only negligible raw-maximum cases (<=1e-6) from that rank diagnostic and report its denominator. Never filter the primary error analysis by correctness, support, effect size, coverage or performance. Positive scaling cannot improve rankings; report rank utility separately. Omitted-source effects and known support annotations are descriptive diagnostics, not causal explained fractions.

## Primary analysis and acceptance rules

Average each test question's paired error difference across the two fixed readers before bootstrapping question identities. Resample within declared hop/type strata using 5,000 draws, seed 20261124, and two-sided 95% intervals. Intervals condition on the fixed development-fitted scales and these fixed readers; they do not include resampling the development-fitting procedure or a population of readers.

H1 is supported only if the upper interval is below zero for all four primary calibrated frozen-minus-recomputed contrasts (two benchmarks x two selectors) at 512 tokens. H2 is supported only if the upper interval is below zero for all four rescaled frozen-minus-zero contrasts. Report unmet criteria honestly, all individual model results, all budgets and incomplete shards. The conjunctions are fixed global claims; individual-cell and non-512 intervals remain secondary/unadjusted. New test data are not reused to choose fitting rules, thresholds, models, budgets or metrics. A separate raw fallback method would require another prospective confirmation cohort.

## Execution and bounded resources

Keep at most one project GPU active through explicit dependencies. Each model's full jobs are gated on its own actual-model smoke. Smokes cover the first hash-selected test question and the longest serialized test context for each benchmark at all four budgets. Use A100 normal queue for both readers, one GPU and 64 GiB host memory per allocation. Smoke wall limit is 45 minutes; each full shard has a 3-hour limit. The base GPU reservation is at most 97 hours 30 minutes; at most two time-limit requeues give a worst-case bound of 292 hours 30 minutes. Process retries occur inside an allocation and do not extend that allocation's limit. Node-failure requeue behavior remains subject to ARC policy.

Atomic per-question checkpoints, exclusive process locks and exact data/source/runtime manifest checks permit restart without mixing experiments. Allow at most three process attempts per allocation. Request a pre-limit signal to checkpoint/requeue; preserve complete questions and recompute only an interrupted question. A source-frozen CPU review with explicit PYTHONPATH follows every GPU outcome and reports missing jobs rather than scoring partial cohorts as complete. No recurring Codex monitor or laptop-dependent process is used. Infrastructure failures remain possible and visible.
