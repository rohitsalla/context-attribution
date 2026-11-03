# Scheduling amendment, before primary-model results

The user requested an earlier GPU start. The original job is 8399008. No 7B results existed when routing was changed.

The job retains its identity and submission time. Routing changes from A100 to H200, the time request decreases from two hours to 30 minutes, CPU request from eight to one, and host memory from 48 GB to 32 GB. GPU count remains one. Model, checkpoint revision, dataset, ordering, decoding, masks, regularization, and evaluation criteria are unchanged. GPU execution remains BF16; H200 timing must be labeled separately from local MLX timing.

The runner now constrains PyTorch CPU threads to the granted Slurm CPU count. Resume is automatic in Slurm and explicit with `--resume` elsewhere. Source/config/dataset hashes must agree before completed conditions can be reused. Each completed condition is flushed and fsynced. A cut-off in an unfinished condition loses that condition's work; it will be rerun. A malformed partial JSONL line causes a failure for inspection rather than silent data deletion.

Up to three serial 30-minute continuation tasks resume after the original allocation, keeping the original two-hour maximum GPU allocation budget. Each skips completed conditions. If all conditions are complete, it reconstructs aggregate reports and exits before loading the model. There is one active pilot GPU at a time. Six software tests include interruption/resume and mismatch rejection checks.

Alternate B200 job 1359559 and preemptible L40S job 747803 were canceled after their actual start estimates proved later. Test-only scheduler probes did not allocate GPUs.
