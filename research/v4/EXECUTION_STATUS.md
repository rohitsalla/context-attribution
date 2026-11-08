# V4 execution receipt

Dated 8 November 2026. This file records a snapshot, not a live monitor.

All V3 jobs completed with exit code zero. The result review is in `RESULTS_REVIEW.md`; the paired-effect method has no demonstrated efficacy win. The initial-state matches and warm-up mismatches are preserved in the result record rather than suppressed.

| Stage | Job | State at this update |
|---|---|---|
| Exact-state, repeated-gradient and lexical-identity smoke | 8548538, task 0 | Completed, exit 0, 33 seconds |
| Evidence/oracle studies and stronger 7B compressor controls | 8548542, tasks 5–13, concurrency 1 | Task 5 running on H200; remaining tasks queued |
| Exact-state pure-soft audit and three anchored seeds | 8548544, tasks 1–4, concurrency 1 | Queued after the evidence array |
| CPU aggregation | 8548545 | Queued after the learning array |

Frozen runtime source is commit `d20dd98`. All 29 tests pass locally and in ARC's actual environment. On the real-HF smoke, same-state gradient hashes agree, source-local cached/cold embeddings agree, initial lexical inputs are exactly identical, and branch start hashes agree for pool parameters, full AdamW state, CPU RNG and CUDA RNG. These are implementation checks on two examples/two updates, not efficacy results.

The first full Hotpot/Qwen evidence task is writing raw, equal-source lexical, global lexical and explicitly annotation-informed oracle outputs. All 96-question MuSiQue and Hotpot tasks, and the learning comparisons, remain exploratory development. A limited cross-benchmark audit found zero exact case-folded question matches with the Hotpot train/dev subset and 11 shared document titles; this is not a pretraining-contamination test or a claim of corpus independence.

The dependency chain allows at most one GPU. The maximum new allocation is 5.5 GPU-hours (five-minute smoke plus thirteen 25-minute tasks). Timeouts stay visible as incomplete studies. CPU aggregation runs automatically after the arrays finish; no locked-test example is loaded. The full runtime, model/data/source hashes, traces and checkpoint receipts stay with the ARC outputs.
