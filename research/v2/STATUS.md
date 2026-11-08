# V2 execution status

Updated 8 November 2026, 13:15 Eastern. This file is a dated receipt, not a live monitor.

## Implemented and verified locally

- Revision-pinned HotpotQA: 512 training, 96 development and 400 locked-test questions. One malformed training annotation is retained and flagged; supporting-fact labels are not method inputs.
- Five hard-memory systems, source/group dependency contracts, fixed answer-token scoring, actual achieved budgets, cached/cold replay comparisons, raw-reference effect distortion, and a no-context diagnostic.
- Frozen-HF source-local soft memory and matched factual-only/paired-effect optimization canaries, with initial-parameter hashes and gradients recorded.
- Literature audit, formal targets/proofs with limitations, experiment gates, paper outline and execution guide.
- All 20 local tests pass. Actual HF GPU execution has **not yet passed** the queued preflight.

## ARC jobs submitted

| Stage | Job | State at this update |
|---|---|---|
| One-question real-HF execution preflight | 8545538 | Pending: priority |
| Matched training controls, array tasks 4–5 | 8545542 | Depends on successful preflight |
| Four model/compressor profiles, array tasks 0–3 | 8545543 | Depends on successful training canaries |
| Descriptive result aggregation | 8545544 | After profile array finishes |

The hard-memory matrix comprises 480 compressed conditions and 32 raw references (512 records) across the four profiles. These are planned counts, not completed results. Each profile uses eight development questions at three budgets. Each training canary uses eight **training** questions for 64 steps. No test-model evaluation has run.

At most one GPU runs concurrently. The allocation is capped at 10 minutes for preflight and 55 minutes per each of six tasks (5 hours 40 minutes total if all time limits are reached). This cap may leave incomplete profiles, which the summary reports explicitly; it does not guarantee matrix completion. Hard-memory traces resume only with unchanged fingerprints. The scheduler currently estimates preflight at approximately **9 November, 03:00 Eastern**, subject to change. Dry-run A100 and preemptible estimates were later; no duplicate GPU jobs were launched.

## Interpretation and remaining work

There are no V2 QA, intervention-fidelity or training efficacy results yet. The v1 study remains a completed negative pilot for its claimed efficiency advantage. Literature overlap rules out presenting recursive provenance, caching, pooling/projection, or factual distillation alone as novel.

Next decisions depend on actual development outcomes: validate fixed-target/cold replay, inspect achieved budgets and task quality, then train and evaluate the paired-effect method against factual-only and unpaired perturbation controls. Full 96-question development, global soft-memory and official ContextCite/AttriBoT baselines, multiple seeds, a second natural benchmark and independent review precede locked-test evaluation. These are outstanding work, not completed paper claims. No ICML readiness or acceptance is asserted.
