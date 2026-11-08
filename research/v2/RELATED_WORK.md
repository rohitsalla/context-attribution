# Related-work audit, 8 November 2026

This is a targeted audit of the closest work, not an exhaustive novelty certificate. Paper-specific descriptions below are based on primary arXiv pages/full text and official repositories. The v1 negative findings remain in `RESULTS.md`.

| Work | Established contribution | Consequence for this project |
|---|---|---|
| [ContextCite, 2409.00729](https://arxiv.org/abs/2409.00729), [official code](https://github.com/MadryLab/context-cite) | Context ablations and sparse surrogate attribution. | Use the official implementation for the paper baseline; the existing custom approximation cannot be relabeled a reproduction. |
| [AttriBoT, 2411.15102](https://arxiv.org/abs/2411.15102), [official code](https://github.com/r-three/AttriBoT) | Efficient LOO through activation caching, hierarchy, and proxy models. | Caching or small-model screening is not itself a novel contribution. Compare optimized scoring baselines, not artificially slow full replay. |
| [Tokengeist, 2607.22610](https://arxiv.org/abs/2607.22610) | Recursive multi-turn provenance and dependency paths. Its evaluated backends are correlational; counterfactual integration is future work. | Do not claim recursive provenance as new. Provenance paths and executed source effects are separate targets. |
| [Causal Agent Replay, 2606.08275](https://arxiv.org/abs/2606.08275) | Executed interventions, run-forward estimators, and interaction-aware credit. | Replay and conditional causal effects already have explicit precedent. |
| [FOCUS, 2609.37590](https://arxiv.org/abs/2609.37590) | Test-time compression guided by draft future plans, dependency estimates, and defensive verification. | Decision-preserving or causally motivated compression is not a new general framing. Compare on task quality and actual computation. |
| [Attribution-Compression Frontier, 2609.14245](https://arxiv.org/abs/2609.14245) | Emitted versus grounded citation evaluation across compressors; distinguishes nominal from achieved compression. | Citation degradation after compression is already studied. Our target must be executed interventions and measured effect preservation, not another citation-only comparison. |
| [REMORY, 2610.11287](https://arxiv.org/abs/2610.11287) | Learned soft residual memory supplements summaries to approximate full-history continuations. | Soft memory plus factual continuation distillation is established. The candidate distinction is source-separable intervention semantics and paired-effect training, subject to empirical validation. |
| [AutoCompressors, 2305.14788](https://arxiv.org/abs/2305.14788), [Gist Tokens, 2304.08467](https://arxiv.org/abs/2304.08467) | Learned compact context representations and reusable soft prompts. | Pooling, projection, learned slots, and reuse are architecture baselines, not inventions here. |
| [RECOMP, 2310.04408](https://arxiv.org/abs/2310.04408) | Task-trained extractive/abstractive RAG compression and selective augmentation. | Include trained compression baselines; preserving factual QA alone is not the proposed new objective. |
| [AtP*, 2403.00745](https://arxiv.org/abs/2403.00745) | Approximate screening followed by intervention verification, with explicit false-negative analysis. | Rank-then-verify and screening-plus-verification are not novel in general. |
| [Minimal Repair Families, 2608.29228](https://arxiv.org/abs/2608.29228) | Joint counterfactual replay to distinguish interaction-dependent repairs. | Duplicate evidence and group interventions need explicit controls; do not claim general interaction recovery from singleton deletion. |
| [Replay reporting standard, 2611.05650](https://arxiv.org/abs/2611.05650) | Measured restoration/replay fidelity and estimand-matched evidence. | Record actual input receipts, unchanged controls, and independently recomputed branches. Same seeds or hash-identical snapshots alone are insufficient evidence. |
| [In-weight overlap and attribution, 2607.23804](https://arxiv.org/abs/2607.23804) | Examines context attribution when information also exists in model weights. | Treat source effects as pipeline-conditional, not as unique knowledge origin. Add no-context and pseudonymized diagnostics before broad claims. |

## Candidate contribution boundary

The provisional direction is **intervention-faithful memory compression**: measure whether compression preserves a declared reference's intervention contrasts, and train a source-separable compressor with a paired-effect objective so audits can reuse unaffected components. The relevant comparison is the same architecture with factual-only training, plus strong globally compressed and extractive baselines.

The structural commutation fact is elementary incremental computation. It is not an ICML contribution on its own, and does not prove preservation of the raw-context model's behavior. A credible paper requires a nontrivial empirical improvement in the quality/effect-fidelity/cost frontier, or a strong, sufficiently general negative finding. No first-of-its-kind or acceptance claim is made.

## Required baselines before a paper claim

Official ContextCite and optimized AttriBoT where compatible; source-ID and lexical controls; raw context; per-source lead and lexical compression; global abstractive compression; global soft compression; identical source-local soft architecture trained factually; compute-matched factual training on perturbed contexts; paired-effect training. FOCUS/REMORY/RECOMP comparisons require compatible released implementations/checkpoints or a clearly labeled reproduction. Listing a paper is not evidence that its method has been run.
