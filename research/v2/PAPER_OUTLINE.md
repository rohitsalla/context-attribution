# Provisional manuscript: Intervention-Faithful Memory Compression

Status: research design and implementation scaffold, **not a completed paper**. New method efficacy has not been established. Keep the v1 negative result visible rather than rewriting it as a successful method.

## Abstract content, after the evidence exists

State the distinction between factual QA/citation consistency and preservation of executed source effects. Define source-separable memory and the paired training objective. Report quality-matched auditing/fidelity results on real benchmarks and multiple HF model families. Include a numerical result only after the corresponding gate passes; otherwise state the negative finding. Do not substitute the old pilot numbers for new-method results.

## 1. Introduction

Motivation: compressed agent memory can preserve an answer while changing which source interventions matter. Existing provenance paths, citation entailment, and replay frameworks answer different questions. Proposed target: an empirical quality/effect-preservation/audit-cost frontier. The co-designed system changes the answering pipeline; it is not a more accurate estimator for an unchanged global summarizer by definition.

## 2. Related work and scope

Use `RELATED_WORK.md`. Treat compression, soft memory, replay, and caching as prior art. Explain specifically what the paired-effect objective and source-local contract add if they outperform the strongest controls. Include the newest attribution-compression and replay-certificate papers.

## 3. Problem formulation and basic propositions

Own-pipeline effects versus raw-reference effects; fixed target tokens; source interventions; fixed routing/budgets; deterministic assumptions. Present the commutation and affected-component arguments as design facts. Present the factual-agreement counterexample and classical transfer bound with explicit assumptions. Do not market elementary algebra as the main novelty.

## 4. Method

Source-separable soft memory, frozen encoder/answerer, attention pooling/projection, paired-effect training, and matched factual/unpaired controls. State which components are actually implemented/trained. The current hard-text systems are diagnostic baselines. Global soft memory and learned routing are not implemented yet and must not appear as completed experiments.

## 5. Experimental design

Natural multi-hop QA; a second memory/QA benchmark; immutable dataset/model revisions; development versus locked test; achieved budgets; QA, causal-effect distortion, replay receipts, and cost; sample accounting; uncertainty; human review scope. Include no-context and entity-aliasing diagnostics for parametric knowledge.

## 6. Results

- Table 1: QA EM/F1 and actual memory positions for all systems.
- Figure 1: QA/effect-preservation/cost frontier at matched resource constraints.
- Table 2: factual-only versus paired-effect versus unpaired augmentation, same architecture/compute.
- Figure 2: raw-reference effect error by compression level, source count and model family.
- Table 3: actual audit replay receipts, errors, costs and failure counts.
- Ablations: source locality, fixed budgets, routing granularity, paired loss, source slots and target spans.

All cells are pending new results. The v1 result belongs in motivation or a separate pilot appendix, with its limitations.

## 7. Discussion and limitations

No unique causal-credit guarantee, no hidden equality to raw-context behavior, no universal confidence certificate from finite replay controls, no claim that factual answer correctness establishes effect preservation. Discuss representation changes, dataset contamination/parametric overlap, corpus overlap, annotator scope, hardware effects and training amortization.

## Appendices and artifact

Proofs, prompts, exact HF revisions, dataset licenses, source hashes, source routing/budget policies, all failures, matched training schedules, extra seeds and independent replay receipts. Never fill a results table with extrapolated or aspirational numbers.
