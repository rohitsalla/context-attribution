# V2 development protocol, fixed before new-model outcomes

## Research direction and limits

Working title: **Intervention-Faithful Memory Compression**. The study co-designs a compressor and an auditable memory representation. It does not claim to improve attribution of an unchanged global summarizer by changing that summarizer. Cached-versus-recomputed equality is an own-pipeline property; raw-reference effect preservation and QA quality are separate measurements.

The v1 completed experiment remains a negative pilot for the proposed efficiency advantage over source-ID tracing. Its test results are not new development data for v2. Related work and the limits of the novelty claim are recorded in `RELATED_WORK.md`.

## Data and leakage controls

Use HF HotpotQA distractor at revision `1908d6afbbead072334abe2965f91bd2709910ab`. Freeze 512 training questions and 96 development questions by a predetermined hash ordering of one pinned training shard. Freeze 400 official-validation questions as a locked test. Question IDs are disjoint; document-title overlap is reported because the retrieval corpus is shared. Preserve duplicate titles and flag out-of-range support annotations instead of silently repairing them. Gold answers are used only for training supervision and evaluation; gold supporting facts are never inputs to compression, routing, or selection.

The first feasibility run uses only eight development questions per model. This is an implementation and direction check, not a powered benchmark. No locked-test model evaluation is enabled in the feasibility runner. A second benchmark (MuSiQue or LongMemEval, after official provenance/format review) and an independently reviewed natural-text sample are required before final paper claims. Neither is claimed implemented here.

## Models

Pinned HF answerers: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3. Qwen2.5-1.5B-Instruct is the initial hard-compression proxy. A Qwen2.5-7B self-compressor condition separates small-compressor failure from representation failure. Revisions and model-card links are in `models.json`. Use existing lawfully available Llama weights; do not attempt to bypass model access conditions. No hosted paid model API is used.

## Hard-memory feasibility matrix

For each of the four model/compressor profiles, evaluate eight dev questions, budgets 128/256/512 answerer-memory tokens, and five memory systems: global abstractive, source-local abstractive, four fixed hash components, per-source lead, and per-source lexical selection. Evaluate raw context once per question. That is 128 records per profile, 512 across four profiles. Hash grouping is fixed by source ID and seed; it is not refit after interventions.

Token-space framing counts against the memory budget. Per-component caps are fixed from the original universe and remain fixed after deletion. Generated text is re-tokenized for the answerer and clipped to the declared quota. Report actual input positions, achieved memory ratio, clipping, and framing overhead; a nominal cap that never binds cannot support a budget-sensitivity claim. Do not lengthen prompts artificially just to obtain a positive result.

For every method, generate the original answer once. Singleton audit branches regenerate only affected compressor components and score the original token sequence; **they do not generate a new answer merely to measure its likelihood**. Record common original costs and actual complete answer-plus-audit costs. The complete global baseline uses the same optimized likelihood-only intervention path. Randomize method order per question/budget. Use synchronized GPU timing and count generation/scoring input and output tokens separately.

Each condition also has an unchanged compaction replay and two ID-selected independently recomputed deletion branches. Record actual token streams/input hashes, score differences, and reference scopes. Cold-replay validation and common-raw-target scoring are diagnostic overhead and are separately charged in the research record, outside the production answer-plus-audit comparison. No empirical certification is claimed for unsampled branches beyond the explicit structural/determinism assumptions.

## Metrics and estimands

Report Hotpot-style answer EM/F1, raw/no-context behavior, achieved token usage, original-plus-audit wall time, and model calls. Own-pipeline replay fidelity is tested against cold recomputation. Compare raw and compressed intervention vectors on the **same raw-answer token sequence**; report absolute error, normalized error with reference scale, negligible-effect counts, and rank recovery. Do not compare effects on different answer strings as if they had the same estimand. Conditional intervention influence is not unique causal credit or knowledge origin.

The frozen-summary citation-removal baseline is evaluated only against the corresponding global pipeline's own executed effects. Missing source mappings are recorded. Full raw-context attribution, lead compression, and lexical compression are essential cheap controls. The current feasibility code does not implement official ContextCite/AttriBoT; those remain mandatory before an efficiency paper comparison.

## Soft-memory objective canary

Run matched factual-only (lambda_cf=0) and paired-effect (lambda_cf=1) canaries with the same eight training examples, initial seed, source encoder, eight slots/source, 64 optimizer steps, teacher computations, and student perturbation passes. Only pooling/projection parameters train; the HF encoder and answerer are frozen. Record initial parameter hashes, finite gradients, initial/final training loss, source truncation, and timing. A falling training loss is **not evidence of improved held-out QA or attribution**.

## Gates toward a paper

1. **Implementation:** all contract tests pass; actual unchanged/cold receipts agree; fixed target scoring is correct; source/budget/model fingerprints protect resumes.
2. **Development viability:** on all 96 dev questions, identify at least two genuinely different achieved memory budgets. At a useful budget the proposed representation should retain QA close to the strongest baseline. Eight-case feasibility means cannot establish noninferiority.
3. **Learning contribution:** after factual warm-up, paired-effect training must improve raw-reference intervention distortion over the identical factual-only architecture and a compute-matched unpaired perturbation baseline. Preliminary target: at least 20% relative distortion reduction, with no more than two QA-F1 points lost. These are development targets, not observed results or guarantees.
4. **Cost frontier:** include initial encoding and audit cost, stronger cached/optimized baselines, actual online timings, and training amortization. A cheaper but worse answering system or a win only against random selection is not sufficient.
5. **Confirmation:** freeze architecture/hyperparameters; evaluate the 400 locked questions and a second natural benchmark across three model families and multiple training seeds. Use paired question-level intervals, family/dataset slices, failures and incorrect-answer slices. Include factual-only, global soft-memory, unpaired perturbation-distillation, and official attribution baselines.
6. **Novelty review:** directly compare the final method with FOCUS, REMORY, RECOMP, attribution-compression evaluation, and incremental-computation predecessors. The elementary commutation proposition alone cannot support a top-tier claim. No ICML acceptance or readiness is promised before these gates.

GPU policy: one concurrent GPU; a short execution canary precedes four model/compressor profiles and two training canaries. Bound the initial development allocation to roughly six GPU-hours. Jobs must exit for missing weights or mismatched fingerprints rather than hold an idle GPU. Further scaling requires the development gates, not a presumption that the idea succeeds.
