# Development hypothesis review

These are development observations. Eight-question pilot means are unstable; they are not powered paper results. The locked test remains untouched.

| Answerer/profile | System | Budget | QA F1 | Effect MAE (sum log p) | Audit time / raw |
|---|---|---:|---:|---:|---:|
| profile0 | raw | full | 55.0 | 0 | 1 |
| profile0 | global | 512 | 62.5 | 7.214 | 21.95 |
| profile0 | source | 512 | 37.5 | 3.363 | 7.06 |
| profile0 | group4 | 512 | 50.0 | 4.230 | 13.42 |
| profile0 | lead | 512 | 62.5 | 2.018 | 0.46 |
| profile0 | lexical | 512 | 72.5 | 2.214 | 0.47 |
| profile1 | raw | full | 87.5 | 0 | 1 |
| profile1 | global | 512 | 62.5 | 5.093 | 21.42 |
| profile1 | source | 512 | 61.1 | 1.286 | 6.93 |
| profile1 | group4 | 512 | 37.5 | 2.810 | 12.91 |
| profile1 | lead | 512 | 62.5 | 1.245 | 0.50 |
| profile1 | lexical | 512 | 88.6 | 0.573 | 0.53 |
| profile2 | raw | full | 68.3 | 0 | 1 |
| profile2 | global | 512 | 50.0 | 4.892 | 19.17 |
| profile2 | source | 512 | 40.7 | 3.010 | 5.92 |
| profile2 | group4 | 512 | 42.4 | 5.724 | 12.87 |
| profile2 | lead | 512 | 50.0 | 2.408 | 0.52 |
| profile2 | lexical | 512 | 79.6 | 2.027 | 0.45 |

## Interpretation

The caching contract is passing sampled replay checks. The initial source-local abstractive method does not yet win the quality/effect/cost comparison against lexical selection.

Correct factual answers can coexist with changed intervention rankings. This motivates testing an intervention-preserving objective, but does not establish novelty or effectiveness. The soft-memory canaries establish optimization only. Larger matched factual/unpaired/paired controls and untouched-question evaluation are required.

Nominal budgets are upper bounds: generated summaries often use fewer positions than extractive memory. Do not attribute every observed difference to source locality or compare summed-log-probability MAE across answerer families as if target lengths were equal. V3 adds mean-log-probability effects, a zero-effect predictor, joint deletions and position-matched lexical controls.

Full machine-readable follow-up results, including incomplete flags and question-level paired intervals, are in the adjacent JSON. Training success is not held-out efficacy.
