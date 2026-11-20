# Four-reader, three-dataset generalization study

This next exploration follows the completed Hotpot confirmation and the secondary RECOMP result. The main question is whether the fidelity benefit of frozen deletion extends across reader size, model family and benchmark. All experimental choices are fixed before new model scores.

## Matrix and data

Run Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 at the pinned revisions in `GENERALIZATION_MODELS.json`. The two existing readers provide continuity, Mistral adds a third family, and Qwen14 adds a within-family scale comparison.

Use the original 400 Hotpot questions, 240 fresh MuSiQue questions (80 per hop count), and 200 2WikiMultiHopQA questions (50 per question type). The fresh cohorts are sampled by a fixed ID hash, excluding normalized questions from all previous project cohorts. Dataset revisions, raw hashes, output hashes, source licenses and mirror limitations are recorded in `data/generalization`. MuSiQue's official authors note that single-hop seeds can overlap Natural Questions training. The NQ-trained RECOMP checkpoint can therefore have related source exposure. The cohort checks cover question overlap and do not constitute pretraining or document contamination audits.

The 400 Hotpot questions have already been evaluated on the original readers. Treat those cells as replication under the new runner and retain their prior-confirmation status in the narrative. The fresh benchmark cohorts test transfer of a hypothesis developed on Hotpot. New data come from public official validation/development subsets through revision-pinned mirrors.

## Shared prediction and intervention contracts

Each question has one raw greedy factual answer and fixed target tokens. Score every singleton source deletion and two source-pair deletions chosen by a fixed ID hash before scoring. Every policy uses those same targets and raw references. Use 512 answerer-token caps, the existing provenance-preserving sentence frames, source order, and greedy budget packing. The five policies are BM25 frozen deletion, BM25 fixed-ranking repacking, BM25 refitted ranking/repacking, RECOMP frozen deletion and RECOMP repacking using independent cached sentence scores. Source and sentence ownership is explicit.

Use FP32 reader weights and encoder weights, forced SDPA math attention, TF32 disabled, explicit 4D causal masks, compact positions and double log-softmax. The scorer requests only the final target-span logits to reduce peak memory. Verify this model interface in preflight and in actual-model smokes. Check repeated raw factual scores and one re-executed raw deletion branch per question within 1e-7. Store immutable source, model and dataset hashes, all effects and positions, generation caps and runtime.

Generate compressed factual answers once per selector. As a decision-level diagnostic, generate a deletion answer under all five policies and raw context for the source with largest absolute raw effect. That target source is deliberately selected for diagnostic severity. Report these diagnostics with their selection rule rather than as prevalence over arbitrary source removals.

## Analysis and claim rule

For each model and dataset, report per-question MAE and pooled NMAE against raw effects for singleton and joint deletions, the zero-effect baseline, top-effect recovery, factual QA and achieved lengths. Report all five policies, including adverse results. Rank metrics exclude negligible raw effects under the established comparison function. The pair diagnostic covers only two predetermined pairs per question.

The primary dataset-level analysis averages each question's paired error differences over the four fixed readers, then bootstraps question IDs. This clusters repeated readers on shared questions. Apply the frozen-minus-recomputed contrast to BM25 and frozen-minus-repacked to RECOMP, giving six dataset/selector contrasts. The overall generalization criterion requires the upper two-sided 95% interval below zero for all six. That conjunction describes the mean of these four readers, while individual-model intervals describe heterogeneity. Use 5,000 bootstrap draws at seed 20261120. Report per-cell intervals as exploratory and unadjusted. Keep raw-reference fidelity distinct from correctness, citation entailment and source ownership.

Primary scores use all questions, with exactly the planned cohort and no model-performance filtering. Failed shards and memory/time-limit errors remain visible as incomplete. Any technical retry preserves question membership, target policy and metrics.

## Execution reservation

Four model-specific smokes precede their corresponding full jobs. Each smoke covers a deterministic first and long serialized-context case on each dataset. The full matrix consists of eight stride shards per dataset and reader, totaling 96 jobs and 3,360 reader–question evaluations. Keep at most one project GPU active by chaining jobs explicitly. The three smaller readers use A100 and the 14B reader uses H200 to allow more FP32 attention memory on long prompts. Each full job reserves at most 90 minutes, and each smoke reserves at most 20 minutes, giving a maximum reservation of 145 hours 20 minutes. Actual usage is expected to be lower and is recorded by Slurm. CPU review follows every outcome. A completion monitor reports meaningful failures and the final verified results.
