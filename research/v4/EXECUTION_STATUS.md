# V4 execution receipt

Dated 8 November 2026. This file records a snapshot, not a live monitor.

All V3 jobs completed with exit code zero. The result review is in `RESULTS_REVIEW.md`; the paired-effect method has no demonstrated efficacy win. The initial-state matches and warm-up mismatches are preserved in the result record rather than suppressed.

| Stage | Job | State at this update |
|---|---|---|
| Exact-state, repeated-gradient and lexical-identity smoke | 8548538, task 0 | Completed, exit 0, 33 seconds |
| First Qwen/Hotpot evidence/oracle study | 8548542, task 5 | Completed: all 384 records |
| Three anchored seeds, all five objectives per seed | 8548544, tasks 2–4 | Completed, exit 0; exact shared states pass |
| Original positional intervention diagnostic | 8548641 | Completed, invalid packed-mask semantics |
| Explicit-mask positional repair | 8549582 | Completed; BF16 equivalence remains outside tolerance |
| Global extraction, anchor allowance and cap 512 | 8549468; 8549583 | Both completed on identical runtime/reference |
| Selected CUDA BF16/FP32 and 2D/4D validation | 8549800 | Queued after remaining evidence work |
| Worst selected case, scheduled CPU FP32 validation | 8550023 | Completed; 2D/4D equality and dense/sparse tolerance pass |
| Pure-soft single-seed audit | 8548544, task 1 | Deferred/cancelled before allocation; resources reallocated |
| Remaining evidence/oracle and stronger compressor studies | 8548542, tasks 8–13 | MuSiQue/strong-compressor stages continuing; Hotpot tasks 5–7 complete |
| CPU aggregation | 8548545 | Queued after the remaining evidence array |

Frozen runtime source is commit `d20dd98`. All 29 tests pass locally and in ARC's actual environment. On the real-HF smoke, same-state gradient hashes agree, source-local cached/cold embeddings agree, initial lexical inputs are exactly identical, and branch start hashes agree for pool parameters, full AdamW state, CPU RNG and CUDA RNG. These are implementation checks on two examples/two updates, not efficacy results.

All three 96-question Hotpot evidence studies and all anchored suites are complete. See `ANCHORED_RESULTS.md`, `ANCHORED_REVIEW.json`, and the three model-specific evidence records. The paired objective offers only a modest improvement over factual-only and no reliable unique advantage over unpaired supervision. The 353-to-506-position same-runtime global-extraction control reduces effect MAE substantially, with uncertain mean QA change. The other-model/MuSiQue and stronger-compressor work remains exploratory development. A limited cross-benchmark audit found zero exact case-folded question matches with the Hotpot train/dev subset and 11 shared document titles; this is not a pretraining-contamination test or a claim of corpus independence.

The original position source is `04a7e48`, repair is `c7eb11b`, CUDA precision source is `a6d7042`, and the separately copied CPU variant is `f18bc1e`; the frozen V4 runtime is unchanged. The CPU check covers only one deliberately selected worst failure: its largest dense/sparse FP32 error is 0.0000513 mean log probability and its 2D/4D difference is zero. It does not certify CUDA behavior or the full cohort. The BF16 alternate-position aggregate remains unvalidated.

All 31 local tests pass; the original 29 core tests also passed on ARC. The dependency chain allows at most one project GPU. The maximum stage allocation is 5 hours 59 minutes after deferring the unused pure audit; the CPU check separately has a 15-minute/32-CPU limit. The older main array was unblocked when the newer precision job lost immediate GPU priority. Timeouts stay visible as incomplete studies. CPU aggregation runs after the evidence array; no locked-test example is loaded. Runtime/model/data/source hashes, traces and checkpoint receipts stay with the ARC outputs.

Latest completion: all three MuSiQue studies and the Qwen7 stronger-compressor pilot have finished; see `MUSIQUE_RESULTS.md`. Stronger-compressor Llama/Mistral tasks are continuing. Conditional full-cohort FP32 positional job **8551169** follows CUDA precision job 8549800 with an after-success dependency and a numerical pass gate. Its source is `fc051b9`; it uses an isolated module copy. The completed-actual-plus-pending-cap budget upper bound including this 30-minute job was 11,823 seconds, below the 21,600-second stage ceiling, with one project GPU at a time. This accounting supersedes the earlier historical submitted-limit total.

Final V4 update: evidence array tasks 5–13, selected CUDA precision 8549800 and aggregation 8548545 all completed exit zero. Conditional full positional 8551169 skipped after a failed numerical gate; it is not a completed full-cohort experiment. See `REMAINING_RESULTS.md` and `REMAINING_REVIEW.json`. Prior stage GPU use totals 7,489 seconds. New FP32 frontier source `5cb4e18` has smoke job 8553085; planned main reservation is four 45-minute jobs after successful smoke, concurrency one. All 33 local tests pass. The H200 smoke scheduling snapshot estimated November 9 01:17:30 cluster local time; estimates can move, and an A100 dry run was substantially later.

Frontier reservation submitted: array **8553116**, tasks 0–3 (Hotpot Qwen/Llama, MuSiQue Qwen/Llama), each 45 minutes and concurrency one, after successful smoke **8553085**. CPU aggregation **8553117** follows all array outcomes and retains incomplete flags. At submission the smoke is pending priority and the main jobs are pending dependency; none is reported as running. Full planned stage upper bound remains 18,589 GPU seconds.

November 8 22:45–22:50 EST recheck: smoke 8553085 remains pending GPU priority; array 8553116 and review 8553117 remain dependency-pending, with no frontier output directory. No FP32 study has run. H200 normal remains the earliest estimated queue among normal/preemptable H200/A100 checks. CPU ownership audit 8553168 completed exit zero in one minute, reconstructing all 576 old BF16 factual memories exactly. Results: `SELECTION_COVERAGE.md`/JSON. Offline frontier reviewer now adds selection-policy and omitted-source diagnostics (source `ef86536`), without changing GPU runtime `5cb4e18`, budgets, validation gates or the planned primary contrasts. All 33 local tests pass.
