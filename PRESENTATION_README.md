# Context attribution under compression — presentation README

**Idea:** When auditing a compressed context, deleting a source and rerunning the selector can introduce replacement evidence. We test whether freezing the original selection gives source effects closer to deleting that source from the full context.

For the same raw greedy answer tokens, measure the change in mean answer-token log probability after each source deletion. Compare compressed effects with that raw-context reference using mean absolute error (MAE; lower is better).

- **Frozen deletion:** remove selected chunks belonging to the deleted source; do not refill.
- **Fixed-ranking refill:** keep candidate priorities fixed and repack surviving evidence.
- **Recomputed selection:** refit ranking/repacking after deletion. RECOMP uses independent sentence scores, so its comparison changes packing.

## Experiments run

| Phase | Scope | Outcome |
| --- | --- | --- |
| Replay prototype | 120 mixed fictional/facts-page cases; 12 online checks | No established efficiency win over cheap source-ID tracing. |
| Abstractive/source-local pilot | Small development profiles | No demonstrated win over strong lexical selection. |
| Paired / anchored training | Three matched-state seeds; 96 development questions | No reliable special advantage over unpaired supervision. |
| FP32 budget frontier | Qwen/Llama; Hotpot 96 + MuSiQue 32 | Larger budgets helped Hotpot; not reliably MuSiQue. |
| Position / replacement controls | Two readers × two benchmarks; 24 questions each | Mechanism unresolved; numerical-equivalence gate failed. |
| Hotpot confirmation + RECOMP follow-up | Same 400 questions; Qwen/Llama | Frozen deletion favored; both refilling and reranking contribute. |
| Four-reader generalization | 4 readers × 840 questions; two selectors; 512-token budget | Complete: 3,360 reader–question evaluations; planned criterion passed. |
| Fresh calibration / budget study | 644 new questions; 2 readers × 4 budgets | Running: 5,152 configurations; calibration tables pending. |

## Completed main results

Four readers: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3 and Qwen2.5-14B. Benchmarks: Hotpot 400, fresh MuSiQue 240 and fresh 2Wiki 200. Use pinned checkpoints, FP32 weights, math attention and fixed raw-answer targets. Hotpot is a reused replication cohort, not fresh confirmation.

**Four-reader averages.** Gain is the reduction in mean singleton attribution MAE. CI is the paired *frozen-minus-recomputed* MAE difference; negative favors frozen. Readers are averaged within question before question bootstrapping.

| Dataset | Selector | Frozen MAE | Recomputed MAE | Error reduction | Paired 95% CI |
| --- | --- | --- | --- | --- | --- |
| Hotpot (400) | BM25 | 0.4705 | 0.6346 | 25.9% | [-0.2308, -0.1116] |
| Hotpot (400) | RECOMP | 0.5018 | 0.5785 | 13.3% | [-0.1074, -0.0499] |
| MuSiQue (240) | BM25 | 0.2816 | 0.3847 | 26.8% | [-0.1450, -0.0694] |
| MuSiQue (240) | RECOMP | 0.2791 | 0.3209 | 13.0% | [-0.0565, -0.0282] |
| 2Wiki (200) | BM25 | 0.4307 | 0.5440 | 20.8% | [-0.1739, -0.0606] |
| 2Wiki (200) | RECOMP | 0.4292 | 0.4799 | 10.6% | [-0.0954, -0.0137] |

All six primary intervals favor frozen deletion. Percentages use the ratio of averaged MAEs, not the average of per-reader percentages.

**Per-reader reductions: BM25 / RECOMP.** These individual intervals are exploratory and unadjusted; † means the RECOMP interval includes zero.

| Reader | Hotpot | MuSiQue | 2Wiki |
| --- | --- | --- | --- |
| Qwen2.5-7B | 24.7% / 11.3% | 26.3% / 9.2% | 13.4% / 8.6%† |
| Llama3.1-8B | 21.2% / 7.9% | 17.0% / 8.6% | 11.1% / 8.9%† |
| Mistral7B-v0.3 | 24.4% / 17.2% | 33.7% / 20.0% | 21.2% / 15.1% |
| Qwen2.5-14B | 29.5% / 14.9% | 28.4% / 16.1% | 27.7% / 10.4%† |

**Interpretation:** Frozen deletion improves fidelity to the specified raw-context deletion reference. It does not establish uniquely correct causal credit or improved answer accuracy. On pooled error, frozen BM25 is worse than predicting zero effects in 6/12 cells; RECOMP in 4/12. Source ranking and deletion-answer agreement improve inconsistently. These absolute-utility gaps motivate the new study.

## Current runs — November 25, 2026, 17:14 EST

Both new reader smokes passed. Completed shards passed source/data/runtime, budget and numerical checks (maximum repeated-score discrepancy 0). Planned time-limit requeues have resumed checkpoints successfully; no unresolved execution errors were found. Counts below include **completed shards only**.

| Reader | Fresh dataset | Complete shards | Dev questions | Test questions | Status |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | musique | 5/8 | 54/84 | 150/240 | Sixth shard running; 18 additional saved questions |
| Qwen2.5-7B | 2wiki | 0/8 | 0/80 | 0/240 | Queued |
| Mistral7B-v0.3 | musique | 0/8 | 0/84 | 0/240 | Queued |
| Mistral7B-v0.3 | 2wiki | 0/8 | 0/80 | 0/240 | Queued |

The sixth Qwen MuSiQue shard is running with 18 additional questions saved. No complete fresh cohort is ready yet. The study has 164 development and 480 test questions: 644 unique questions, 1,288 reader–question evaluations and 5,152 reader–question–budget configurations. Repeated readers/budgets are not independent questions.

## Pending calibration / budget results — intentionally empty

Fit a separate nonnegative effect scale per reader/dataset/budget/policy on development questions only, then evaluate untouched test questions. This is **effect rescaling**, not probability calibration. The 512-token contrasts are primary; other budgets are secondary. Blank cells mean unavailable results, never zero or a claimed failure. No partial-cohort fit is reported. Δ means rescaled frozen-minus-recomputed MAE; negative favors frozen.

| Fresh cohort, two-reader average | Budget | BM25 calibrated Δ MAE [95% CI] | RECOMP calibrated Δ MAE [95% CI] | Frozen vs zero (BM25 / RECOMP) |
| --- | --- | --- | --- | --- |
| MuSiQue | 256 |  |  |  |
| MuSiQue | 512 |  |  |  |
| MuSiQue | 1024 |  |  |  |
| MuSiQue | 2048 |  |  |  |
| 2Wiki | 256 |  |  |  |
| 2Wiki | 512 |  |  |  |
| 2Wiki | 1024 |  |  |  |
| 2Wiki | 2048 |  |  |  |

## Presentation message and next decision

Present a replicated **deletion-protocol effect**. Pairing-specific training, a universal larger-budget solution, and a pure replacement-versus-position mechanism are not established. The next decision is whether the advantage survives development-only rescaling and beats zero prediction in absolute error. Matched-length donor controls are secondary; donors are not neutral padding.

[Full numerical tables, QA, rank metrics and historical controls](research/PRESENTATION_TABLES.md) · [Main aggregate JSON](research/v4/GENERALIZATION_RESULTS.json) · [New frozen protocol](research/v5/STABILITY_PROTOCOL.md) · [New cohort manifest](data/stability/manifest.json) · [Related-work audit](research/v4/LITERATURE_EXPLORATION.md)

For a beginner-facing report narrative, diagrams, verified examples and slide order, use [FINAL_REPORT_DESIGN.md](FINAL_REPORT_DESIGN.md).
