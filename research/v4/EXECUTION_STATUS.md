# V4 execution receipt

Dated 8 November 2026. This file records a snapshot, not a live monitor.

All V3 jobs completed with exit code zero. The result review is in `RESULTS_REVIEW.md`; the paired-effect method has no demonstrated efficacy win. The initial-state matches and warm-up mismatches are preserved in the result record rather than suppressed.

| Stage | Job | State at this update |
|---|---|---|
| Exact-state, repeated-gradient and lexical-identity smoke | 8548538, task 0 | Completed, exit 0, 33 seconds |
| First Qwen/Hotpot evidence/oracle study | 8548542, task 5 | Completed: all 384 records |
| Three anchored seeds and exact-state pure-soft audit | 8548544, tasks 2–4 then 1, concurrency 1 | Task 2 running on H200; remaining tasks queued |
| Positional intervention diagnostic | 8548641 | Queued after learning, ten-minute cap |
| Remaining evidence/oracle and stronger compressor studies | 8548542, tasks 6–13 | Queued after position diagnostic |
| CPU aggregation | 8548545 | Queued after the remaining evidence array |

Frozen runtime source is commit `d20dd98`. All 29 tests pass locally and in ARC's actual environment. On the real-HF smoke, same-state gradient hashes agree, source-local cached/cold embeddings agree, initial lexical inputs are exactly identical, and branch start hashes agree for pool parameters, full AdamW state, CPU RNG and CUDA RNG. These are implementation checks on two examples/two updates, not efficacy results.

The first full Hotpot/Qwen evidence task is complete; see `INTERIM_REVIEW.md`. The repaired anchored comparison has not completed, so no efficacy judgment is available yet. Its shared warm-up and first branch checkpoint have been written. The other model/MuSiQue and learning tasks remain exploratory development. A limited cross-benchmark audit found zero exact case-folded question matches with the Hotpot train/dev subset and 11 shared document titles; this is not a pretraining-contamination test or a claim of corpus independence.

The standalone position probe is separately frozen at commit `04a7e48` and leaves the active V4 runtime unchanged. The new positional-index test passes locally (30 tests total); the original 29 core tests also passed on ARC. The dependency chain allows at most one GPU. The maximum allocation is now 5 hours 40 minutes, including the ten-minute position probe. Timeouts stay visible as incomplete studies. CPU aggregation runs automatically after the arrays finish; no locked-test example is loaded. The full runtime, model/data/source hashes, traces and checkpoint receipts stay with the ARC outputs.
