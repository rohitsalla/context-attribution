# Follow-up execution receipt

Snapshot: 8 November 2026, 15:45 Eastern. This is a dated receipt, not a live status feed.

Completed V2 answerer profiles: Qwen, Llama and Mistral, each 128 records. The Qwen self-compression control (`8545543_3`) is running on an H200. The original eight-example training canaries remain optimization tests, not evidence of a method win.

| Experiment | Job | Status at snapshot |
|---|---|---|
| New soft-memory training smoke | 8546802, task 13 | Completed, exit 0, 23 seconds |
| Preliminary canary evaluation | 8546803, task 0 | Completed, 74 seconds; decoding-confound debugging record, excluded from efficacy |
| Canary evaluation with matched greedy decoding | 8546860 | Queued after the remaining V2 profile |
| Three objectives × three seeds; three expanded cheap-control profiles | 8546804, tasks 1–12, concurrency 1 | Queued after successful matched-decoding check |
| Question-conditioning/alias/discrete-answer mechanism probe | 8546844 | Queued after follow-up array |
| Learning/control report; mechanism report | 8546805; 8546845 | Dependent CPU aggregations |

Learning runtime source: commit `f2189c5` (V2 source remains unchanged). Standalone mechanism module: commit `cd89cb1`. New source directories are isolated from the original running V2 workspace; all outputs carry hashes and model/data revisions. Initial and post-warm-up hashes are checked across objective conditions. All 25 unit tests pass locally and in ARC's actual environment.

The larger learning comparisons and corrected canary have not produced efficacy results at this snapshot. The scheduler dependency chain keeps at most one GPU active. The total follow-up allocation is capped at six additional GPU-hours, including both diagnostic canaries and the mechanism probe. Time limits may produce incomplete profiles; reports retain those flags instead of treating partial output as a successful experiment. No locked-test model evaluation is authorized by these scripts.

The protocol is in `FOLLOWUP_PROTOCOL.md`; observed evidence and the candidate mechanism are in `EVIDENCE_INTERPRETATION.md` and `MECHANISM_PROTOCOL.md`. Official attribution baselines, global soft-memory control, second benchmark and independent validation remain outstanding before a strong paper claim.
