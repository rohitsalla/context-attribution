# Full presentation tables

These tables reproduce the verified main-study aggregate JSON. No pending calibration values are inferred. Main source: b11f39e; repaired CPU review 8951475. The new study remains source b74241d. Numeric precision is rounded for display; use the [aggregate JSON](v4/GENERALIZATION_RESULTS.json) for full precision.

## 1. Primary four-reader paired contrasts

| Dataset | Selector | Frozen-minus-recomputed MAE | 95% question-bootstrap CI |
| --- | --- | --- | --- |
| Hotpot (400) | BM25 | -0.16414 | [-0.23075, -0.11161] |
| Hotpot (400) | RECOMP | -0.07671 | [-0.10738, -0.04989] |
| MuSiQue (240) | BM25 | -0.10310 | [-0.14500, -0.06938] |
| MuSiQue (240) | RECOMP | -0.04182 | [-0.05650, -0.02823] |
| 2Wiki (200) | BM25 | -0.11329 | [-0.17395, -0.06059] |
| 2Wiki (200) | RECOMP | -0.05068 | [-0.09544, -0.01369] |

All six intervals favor frozen deletion. They average fixed readers within each shared question, then resample questions. This is a statement about these readers/datasets, not a population of architectures. Individual reader contrasts below are exploratory/unadjusted.

## Singleton source deletion: Question-average MAE

| Reader | Dataset | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 0.5645 | 0.6898 | 0.7494 | 0.5800 | 0.6539 |
| Qwen2.5-7B | MuSiQue (240) | 0.3850 | 0.4655 | 0.5223 | 0.4089 | 0.4502 |
| Qwen2.5-7B | 2Wiki (200) | 0.4495 | 0.4622 | 0.5193 | 0.5135 | 0.5618 |
| Llama3.1-8B | Hotpot (400) | 0.3113 | 0.3541 | 0.3952 | 0.3310 | 0.3594 |
| Llama3.1-8B | MuSiQue (240) | 0.2124 | 0.2320 | 0.2559 | 0.2147 | 0.2349 |
| Llama3.1-8B | 2Wiki (200) | 0.2414 | 0.2575 | 0.2715 | 0.2734 | 0.3000 |
| Mistral7B-v0.3 | Hotpot (400) | 0.3482 | 0.4175 | 0.4605 | 0.3894 | 0.4706 |
| Mistral7B-v0.3 | MuSiQue (240) | 0.1959 | 0.2543 | 0.2954 | 0.1964 | 0.2454 |
| Mistral7B-v0.3 | 2Wiki (200) | 0.3745 | 0.4521 | 0.4753 | 0.3255 | 0.3832 |
| Qwen2.5-14B | Hotpot (400) | 0.6581 | 0.7739 | 0.9335 | 0.7068 | 0.8302 |
| Qwen2.5-14B | MuSiQue (240) | 0.3332 | 0.4011 | 0.4654 | 0.2964 | 0.3532 |
| Qwen2.5-14B | 2Wiki (200) | 0.6575 | 0.8580 | 0.9099 | 0.6045 | 0.6746 |

## Singleton source deletion: Pooled normalized MAE

| Reader | Dataset | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 0.9067 | 1.1088 | 1.2046 | 0.9318 | 1.0508 |
| Qwen2.5-7B | MuSiQue (240) | 1.4254 | 1.7226 | 1.9325 | 1.5092 | 1.6627 |
| Qwen2.5-7B | 2Wiki (200) | 0.8727 | 0.8973 | 1.0081 | 0.9969 | 1.0906 |
| Llama3.1-8B | Hotpot (400) | 0.8508 | 0.9679 | 1.0804 | 0.9037 | 0.9814 |
| Llama3.1-8B | MuSiQue (240) | 1.2439 | 1.3597 | 1.4994 | 1.2565 | 1.3745 |
| Llama3.1-8B | 2Wiki (200) | 0.7958 | 0.8488 | 0.8950 | 0.9010 | 0.9887 |
| Mistral7B-v0.3 | Hotpot (400) | 1.0125 | 1.2139 | 1.3390 | 1.1322 | 1.3681 |
| Mistral7B-v0.3 | MuSiQue (240) | 1.1517 | 1.4951 | 1.7346 | 1.1560 | 1.4397 |
| Mistral7B-v0.3 | 2Wiki (200) | 1.0687 | 1.2902 | 1.3566 | 0.9289 | 1.0938 |
| Qwen2.5-14B | Hotpot (400) | 0.7339 | 0.8629 | 1.0408 | 0.7872 | 0.9241 |
| Qwen2.5-14B | MuSiQue (240) | 1.1093 | 1.3354 | 1.5493 | 0.9868 | 1.1766 |
| Qwen2.5-14B | 2Wiki (200) | 0.7235 | 0.9441 | 1.0012 | 0.6652 | 0.7423 |

The all-zero predictor has pooled NMAE 1 on these nonzero-reference aggregates. Pooled weighting differs from the primary per-question MAE when source counts vary.

## Singleton source deletion: Top-effect recovery (%)

| Reader | Dataset | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 69.25 | 65.25 | 65.50 | 66.50 | 65.50 |
| Qwen2.5-7B | MuSiQue (240) | 35.00 | 35.42 | 37.08 | 32.92 | 34.58 |
| Qwen2.5-7B | 2Wiki (200) | 59.50 | 57.00 | 56.50 | 56.50 | 55.50 |
| Llama3.1-8B | Hotpot (400) | 69.50 | 71.00 | 69.75 | 65.50 | 63.75 |
| Llama3.1-8B | MuSiQue (240) | 34.58 | 32.50 | 31.67 | 31.25 | 29.17 |
| Llama3.1-8B | 2Wiki (200) | 60.00 | 56.00 | 55.50 | 57.00 | 58.00 |
| Mistral7B-v0.3 | Hotpot (400) | 63.29 | 63.04 | 62.78 | 55.95 | 54.43 |
| Mistral7B-v0.3 | MuSiQue (240) | 50.42 | 43.75 | 43.33 | 40.00 | 41.67 |
| Mistral7B-v0.3 | 2Wiki (200) | 62.50 | 58.50 | 56.00 | 64.50 | 57.00 |
| Qwen2.5-14B | Hotpot (400) | 65.66 | 65.16 | 65.16 | 62.16 | 59.90 |
| Qwen2.5-14B | MuSiQue (240) | 32.63 | 31.78 | 30.51 | 38.14 | 34.32 |
| Qwen2.5-14B | 2Wiki (200) | 52.50 | 51.00 | 54.00 | 55.00 | 54.50 |

Rank diagnostics use the source report’s eligibility rule; denominators can differ from the full cohort. Rank recovery is not answer accuracy.

## Two predetermined source-pair deletions per question: Question-average MAE

| Reader | Dataset | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 0.8135 | 0.9731 | 0.9766 | 0.8757 | 0.9735 |
| Qwen2.5-7B | MuSiQue (240) | 0.6179 | 0.7592 | 0.7911 | 0.8077 | 0.9000 |
| Qwen2.5-7B | 2Wiki (200) | 0.6667 | 0.6651 | 0.7380 | 0.8111 | 0.8922 |
| Llama3.1-8B | Hotpot (400) | 0.4913 | 0.5155 | 0.5286 | 0.5317 | 0.5768 |
| Llama3.1-8B | MuSiQue (240) | 0.3566 | 0.4114 | 0.4053 | 0.3820 | 0.4080 |
| Llama3.1-8B | 2Wiki (200) | 0.3930 | 0.4324 | 0.4503 | 0.4470 | 0.4896 |
| Mistral7B-v0.3 | Hotpot (400) | 0.6137 | 0.6402 | 0.6803 | 0.6952 | 0.7994 |
| Mistral7B-v0.3 | MuSiQue (240) | 0.3014 | 0.3913 | 0.4307 | 0.4031 | 0.4867 |
| Mistral7B-v0.3 | 2Wiki (200) | 0.5855 | 0.7061 | 0.7132 | 0.5330 | 0.5557 |
| Qwen2.5-14B | Hotpot (400) | 1.0268 | 1.1622 | 1.1579 | 1.1594 | 1.1979 |
| Qwen2.5-14B | MuSiQue (240) | 0.5451 | 0.6599 | 0.7563 | 0.5078 | 0.5918 |
| Qwen2.5-14B | 2Wiki (200) | 1.0581 | 1.2075 | 1.2621 | 1.0735 | 1.1687 |

## Two predetermined source-pair deletions per question: Pooled normalized MAE

| Reader | Dataset | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 0.6882 | 0.8232 | 0.8262 | 0.7408 | 0.8235 |
| Qwen2.5-7B | MuSiQue (240) | 1.1251 | 1.3824 | 1.4407 | 1.4708 | 1.6388 |
| Qwen2.5-7B | 2Wiki (200) | 0.8043 | 0.8023 | 0.8903 | 0.9785 | 1.0763 |
| Llama3.1-8B | Hotpot (400) | 0.7867 | 0.8256 | 0.8466 | 0.8515 | 0.9237 |
| Llama3.1-8B | MuSiQue (240) | 1.3609 | 1.5702 | 1.5467 | 1.4579 | 1.5571 |
| Llama3.1-8B | 2Wiki (200) | 0.6102 | 0.6714 | 0.6991 | 0.6940 | 0.7601 |
| Mistral7B-v0.3 | Hotpot (400) | 0.9267 | 0.9667 | 1.0274 | 1.0499 | 1.2073 |
| Mistral7B-v0.3 | MuSiQue (240) | 0.9027 | 1.1719 | 1.2899 | 1.2073 | 1.4576 |
| Mistral7B-v0.3 | 2Wiki (200) | 0.8524 | 1.0279 | 1.0382 | 0.7759 | 0.8090 |
| Qwen2.5-14B | Hotpot (400) | 0.6051 | 0.6849 | 0.6824 | 0.6833 | 0.7059 |
| Qwen2.5-14B | MuSiQue (240) | 0.9816 | 1.1883 | 1.3621 | 0.9146 | 1.0658 |
| Qwen2.5-14B | 2Wiki (200) | 0.5844 | 0.6669 | 0.6970 | 0.5929 | 0.6455 |

The all-zero predictor has pooled NMAE 1 on these nonzero-reference aggregates. Pooled weighting differs from the primary per-question MAE when source counts vary.

## Two predetermined source-pair deletions per question: Top-effect recovery (%)

| Reader | Dataset | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 69.69 | 64.25 | 64.77 | 63.21 | 63.47 |
| Qwen2.5-7B | MuSiQue (240) | 54.39 | 56.07 | 58.58 | 53.97 | 51.46 |
| Qwen2.5-7B | 2Wiki (200) | 68.18 | 65.66 | 65.66 | 70.71 | 69.19 |
| Llama3.1-8B | Hotpot (400) | 65.75 | 64.25 | 61.00 | 67.25 | 67.75 |
| Llama3.1-8B | MuSiQue (240) | 53.33 | 56.25 | 53.75 | 52.92 | 52.50 |
| Llama3.1-8B | 2Wiki (200) | 68.00 | 69.00 | 68.00 | 68.00 | 66.50 |
| Mistral7B-v0.3 | Hotpot (400) | 71.43 | 71.17 | 66.75 | 65.97 | 64.42 |
| Mistral7B-v0.3 | MuSiQue (240) | 56.67 | 57.50 | 55.00 | 58.75 | 59.17 |
| Mistral7B-v0.3 | 2Wiki (200) | 69.35 | 65.83 | 63.82 | 66.33 | 63.32 |
| Qwen2.5-14B | Hotpot (400) | 70.45 | 69.60 | 68.75 | 66.19 | 67.33 |
| Qwen2.5-14B | MuSiQue (240) | 59.26 | 61.11 | 65.74 | 58.33 | 59.72 |
| Qwen2.5-14B | 2Wiki (200) | 65.79 | 65.79 | 65.79 | 63.68 | 65.79 |

Rank diagnostics use the source report’s eligibility rule; denominators can differ from the full cohort. Rank recovery is not answer accuracy.

## Factual answer F1 (%)

| Reader | Dataset | Raw | BM25 | RECOMP |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 54.66 | 44.57 | 47.04 |
| Qwen2.5-7B | MuSiQue (240) | 20.94 | 16.64 | 21.50 |
| Qwen2.5-7B | 2Wiki (200) | 32.90 | 32.90 | 31.37 |
| Llama3.1-8B | Hotpot (400) | 60.16 | 52.96 | 53.21 |
| Llama3.1-8B | MuSiQue (240) | 17.76 | 14.98 | 16.64 |
| Llama3.1-8B | 2Wiki (200) | 34.75 | 27.98 | 28.31 |
| Mistral7B-v0.3 | Hotpot (400) | 54.99 | 49.78 | 48.43 |
| Mistral7B-v0.3 | MuSiQue (240) | 18.69 | 14.76 | 17.16 |
| Mistral7B-v0.3 | 2Wiki (200) | 35.53 | 26.30 | 31.59 |
| Qwen2.5-14B | Hotpot (400) | 57.11 | 48.20 | 45.92 |
| Qwen2.5-14B | MuSiQue (240) | 19.52 | 10.45 | 13.27 |
| Qwen2.5-14B | 2Wiki (200) | 41.91 | 33.99 | 34.09 |

Frozen and recomputed policies share the same factual compressed memory and factual answer by construction. These factual QA values are not separate before-deletion gains for frozen versus recomputed policies.

## Mean factual memory positions

| Reader | Dataset | Raw memory | BM25 memory | RECOMP memory |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 1460.4 | 506.2 | 506.5 |
| Qwen2.5-7B | MuSiQue (240) | 2583.1 | 507.3 | 507.1 |
| Qwen2.5-7B | 2Wiki (200) | 1057.5 | 500.4 | 501.5 |
| Llama3.1-8B | Hotpot (400) | 1395.9 | 506.2 | 506.0 |
| Llama3.1-8B | MuSiQue (240) | 2455.1 | 508.1 | 507.1 |
| Llama3.1-8B | 2Wiki (200) | 999.1 | 500.1 | 500.6 |
| Mistral7B-v0.3 | Hotpot (400) | 1621.5 | 504.9 | 504.6 |
| Mistral7B-v0.3 | MuSiQue (240) | 2876.3 | 506.3 | 506.8 |
| Mistral7B-v0.3 | 2Wiki (200) | 1163.5 | 501.3 | 502.4 |
| Qwen2.5-14B | Hotpot (400) | 1460.4 | 506.2 | 506.5 |
| Qwen2.5-14B | MuSiQue (240) | 2583.1 | 507.3 | 507.1 |
| Qwen2.5-14B | 2Wiki (200) | 1057.5 | 500.4 | 501.5 |

Compressed memory is capped at 512 answerer-token positions. These values exclude the surrounding question/chat prompt and answer continuation.

## Deletion-answer agreement with raw (%)

| Reader | Dataset | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 76.25 | 77.50 | 77.00 | 73.00 | 74.50 |
| Qwen2.5-7B | MuSiQue (240) | 67.08 | 69.17 | 68.75 | 67.92 | 66.67 |
| Qwen2.5-7B | 2Wiki (200) | 81.00 | 82.50 | 83.50 | 75.50 | 77.50 |
| Llama3.1-8B | Hotpot (400) | 77.00 | 78.25 | 77.75 | 77.25 | 78.50 |
| Llama3.1-8B | MuSiQue (240) | 79.58 | 78.75 | 79.17 | 78.75 | 78.75 |
| Llama3.1-8B | 2Wiki (200) | 78.00 | 81.00 | 80.50 | 81.00 | 79.50 |
| Mistral7B-v0.3 | Hotpot (400) | 30.25 | 32.00 | 32.50 | 29.00 | 31.00 |
| Mistral7B-v0.3 | MuSiQue (240) | 10.00 | 11.25 | 10.42 | 11.25 | 15.42 |
| Mistral7B-v0.3 | 2Wiki (200) | 28.50 | 31.50 | 29.50 | 23.00 | 26.00 |
| Qwen2.5-14B | Hotpot (400) | 79.50 | 80.75 | 82.25 | 80.25 | 80.50 |
| Qwen2.5-14B | MuSiQue (240) | 85.00 | 86.25 | 86.25 | 82.50 | 83.33 |
| Qwen2.5-14B | 2Wiki (200) | 86.00 | 87.00 | 87.50 | 85.50 | 86.50 |

Diagnostic deletion chooses the source with greatest absolute raw singleton effect. This severity-selected diagnostic is not prevalence over arbitrary omissions.

## Answer F1 after that severity-selected deletion (%)

| Reader | Dataset | Raw | BM25 frozen | BM25 fixed refill | BM25 recomputed | RECOMP frozen | RECOMP repacked |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | 15.84 | 13.99 | 13.81 | 14.29 | 15.70 | 15.93 |
| Qwen2.5-7B | MuSiQue (240) | 12.13 | 7.25 | 6.42 | 6.75 | 9.01 | 9.25 |
| Qwen2.5-7B | 2Wiki (200) | 12.91 | 8.08 | 7.87 | 7.77 | 8.34 | 10.40 |
| Llama3.1-8B | Hotpot (400) | 17.37 | 16.91 | 15.39 | 16.95 | 17.95 | 17.66 |
| Llama3.1-8B | MuSiQue (240) | 6.50 | 5.94 | 6.07 | 5.66 | 7.13 | 6.85 |
| Llama3.1-8B | 2Wiki (200) | 6.27 | 3.76 | 4.38 | 4.20 | 3.57 | 5.50 |
| Mistral7B-v0.3 | Hotpot (400) | 27.85 | 21.83 | 23.24 | 23.18 | 25.98 | 24.85 |
| Mistral7B-v0.3 | MuSiQue (240) | 9.61 | 6.63 | 7.06 | 6.04 | 7.81 | 7.93 |
| Mistral7B-v0.3 | 2Wiki (200) | 19.59 | 13.16 | 14.31 | 14.72 | 15.22 | 14.58 |
| Qwen2.5-14B | Hotpot (400) | 14.81 | 11.98 | 12.27 | 13.25 | 11.06 | 13.22 |
| Qwen2.5-14B | MuSiQue (240) | 6.29 | 3.12 | 2.50 | 2.52 | 4.12 | 3.73 |
| Qwen2.5-14B | 2Wiki (200) | 10.43 | 7.13 | 6.08 | 5.58 | 5.34 | 5.26 |

## All individual-reader paired contrasts

| Reader | Dataset | First minus second | Mean MAE difference | 95% question-bootstrap CI |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | Hotpot (400) | bm25_frozen-minus-bm25_refitted | -0.18488 | [-0.24937, -0.12668] |
| Qwen2.5-7B | Hotpot (400) | recomp_frozen-minus-recomp_repacked | -0.07391 | [-0.12090, -0.02964] |
| Qwen2.5-7B | Hotpot (400) | bm25_fixed_rank-minus-bm25_frozen | 0.12533 | [0.07729, 0.17884] |
| Qwen2.5-7B | Hotpot (400) | bm25_refitted-minus-bm25_fixed_rank | 0.05955 | [0.01998, 0.10832] |
| Qwen2.5-7B | MuSiQue (240) | bm25_frozen-minus-bm25_refitted | -0.13731 | [-0.21121, -0.07954] |
| Qwen2.5-7B | MuSiQue (240) | recomp_frozen-minus-recomp_repacked | -0.04130 | [-0.06646, -0.01787] |
| Qwen2.5-7B | MuSiQue (240) | bm25_fixed_rank-minus-bm25_frozen | 0.08051 | [0.04636, 0.11877] |
| Qwen2.5-7B | MuSiQue (240) | bm25_refitted-minus-bm25_fixed_rank | 0.05680 | [0.01643, 0.10952] |
| Qwen2.5-7B | 2Wiki (200) | bm25_frozen-minus-bm25_refitted | -0.06978 | [-0.13607, -0.01807] |
| Qwen2.5-7B | 2Wiki (200) | recomp_frozen-minus-recomp_repacked | -0.04826 | [-0.11237, 0.00628] |
| Qwen2.5-7B | 2Wiki (200) | bm25_fixed_rank-minus-bm25_frozen | 0.01266 | [-0.01699, 0.04340] |
| Qwen2.5-7B | 2Wiki (200) | bm25_refitted-minus-bm25_fixed_rank | 0.05712 | [0.02012, 0.10870] |
| Llama3.1-8B | Hotpot (400) | bm25_frozen-minus-bm25_refitted | -0.08398 | [-0.12994, -0.04823] |
| Llama3.1-8B | Hotpot (400) | recomp_frozen-minus-recomp_repacked | -0.02840 | [-0.05605, -0.00468] |
| Llama3.1-8B | Hotpot (400) | bm25_fixed_rank-minus-bm25_frozen | 0.04288 | [0.02004, 0.06885] |
| Llama3.1-8B | Hotpot (400) | bm25_refitted-minus-bm25_fixed_rank | 0.04111 | [0.01498, 0.07940] |
| Llama3.1-8B | MuSiQue (240) | bm25_frozen-minus-bm25_refitted | -0.04348 | [-0.06998, -0.02077] |
| Llama3.1-8B | MuSiQue (240) | recomp_frozen-minus-recomp_repacked | -0.02012 | [-0.03394, -0.00770] |
| Llama3.1-8B | MuSiQue (240) | bm25_fixed_rank-minus-bm25_frozen | 0.01965 | [0.00099, 0.04226] |
| Llama3.1-8B | MuSiQue (240) | bm25_refitted-minus-bm25_fixed_rank | 0.02383 | [0.01312, 0.03674] |
| Llama3.1-8B | 2Wiki (200) | bm25_frozen-minus-bm25_refitted | -0.03009 | [-0.05912, -0.00401] |
| Llama3.1-8B | 2Wiki (200) | recomp_frozen-minus-recomp_repacked | -0.02660 | [-0.06263, 0.00504] |
| Llama3.1-8B | 2Wiki (200) | bm25_fixed_rank-minus-bm25_frozen | 0.01606 | [-0.00922, 0.04526] |
| Llama3.1-8B | 2Wiki (200) | bm25_refitted-minus-bm25_fixed_rank | 0.01403 | [0.00310, 0.02830] |
| Mistral7B-v0.3 | Hotpot (400) | bm25_frozen-minus-bm25_refitted | -0.11232 | [-0.17098, -0.06272] |
| Mistral7B-v0.3 | Hotpot (400) | recomp_frozen-minus-recomp_repacked | -0.08115 | [-0.12108, -0.04644] |
| Mistral7B-v0.3 | Hotpot (400) | bm25_fixed_rank-minus-bm25_frozen | 0.06932 | [0.02804, 0.11925] |
| Mistral7B-v0.3 | Hotpot (400) | bm25_refitted-minus-bm25_fixed_rank | 0.04300 | [0.01484, 0.07515] |
| Mistral7B-v0.3 | MuSiQue (240) | bm25_frozen-minus-bm25_refitted | -0.09946 | [-0.13840, -0.06471] |
| Mistral7B-v0.3 | MuSiQue (240) | recomp_frozen-minus-recomp_repacked | -0.04900 | [-0.07971, -0.02176] |
| Mistral7B-v0.3 | MuSiQue (240) | bm25_fixed_rank-minus-bm25_frozen | 0.05842 | [0.03006, 0.09457] |
| Mistral7B-v0.3 | MuSiQue (240) | bm25_refitted-minus-bm25_fixed_rank | 0.04104 | [0.01490, 0.06840] |
| Mistral7B-v0.3 | 2Wiki (200) | bm25_frozen-minus-bm25_refitted | -0.10089 | [-0.16204, -0.04683] |
| Mistral7B-v0.3 | 2Wiki (200) | recomp_frozen-minus-recomp_repacked | -0.05776 | [-0.11418, -0.00728] |
| Mistral7B-v0.3 | 2Wiki (200) | bm25_fixed_rank-minus-bm25_frozen | 0.07761 | [0.02641, 0.13634] |
| Mistral7B-v0.3 | 2Wiki (200) | bm25_refitted-minus-bm25_fixed_rank | 0.02328 | [0.00174, 0.04779] |
| Qwen2.5-14B | Hotpot (400) | bm25_frozen-minus-bm25_refitted | -0.27537 | [-0.44492, -0.15099] |
| Qwen2.5-14B | Hotpot (400) | recomp_frozen-minus-recomp_repacked | -0.12337 | [-0.19336, -0.06557] |
| Qwen2.5-14B | Hotpot (400) | bm25_fixed_rank-minus-bm25_frozen | 0.11581 | [0.05889, 0.18099] |
| Qwen2.5-14B | Hotpot (400) | bm25_refitted-minus-bm25_fixed_rank | 0.15956 | [0.06497, 0.31048] |
| Qwen2.5-14B | MuSiQue (240) | bm25_frozen-minus-bm25_refitted | -0.13217 | [-0.21890, -0.06197] |
| Qwen2.5-14B | MuSiQue (240) | recomp_frozen-minus-recomp_repacked | -0.05687 | [-0.09053, -0.02426] |
| Qwen2.5-14B | MuSiQue (240) | bm25_fixed_rank-minus-bm25_frozen | 0.06791 | [0.02569, 0.11375] |
| Qwen2.5-14B | MuSiQue (240) | bm25_refitted-minus-bm25_fixed_rank | 0.06426 | [0.01490, 0.12650] |
| Qwen2.5-14B | 2Wiki (200) | bm25_frozen-minus-bm25_refitted | -0.25239 | [-0.43529, -0.10354] |
| Qwen2.5-14B | 2Wiki (200) | recomp_frozen-minus-recomp_repacked | -0.07011 | [-0.21616, 0.02381] |
| Qwen2.5-14B | 2Wiki (200) | bm25_fixed_rank-minus-bm25_frozen | 0.20048 | [0.06334, 0.37842] |
| Qwen2.5-14B | 2Wiki (200) | bm25_refitted-minus-bm25_fixed_rank | 0.05191 | [0.00556, 0.10206] |

## Historical: Replay prototype: mixed synthetic/facts-page data

Source and qualifications: [RESULTS.md](../RESULTS.md). These cohorts/targets differ from the main matrix; do not pool them as independent confirmation.

| Scope / replay budget | Recursive local ranking | Source-ID tracing | Random subset expectation |
|---|---:|---:|---:|
| Held-out test, B=1 (80 questions) | 40.0% | 33.8% | 17.7% |
| Held-out test, B=2 | 68.8% | 62.5% | 35.0% |
| Held-out test, B=3 | 78.8% | 73.8% | 51.9% |
| All questions, B=1 (120 questions) | 51.7% | 47.5% | 17.4% |
| All questions, B=2 | 74.2% | 73.3% | 34.4% |
| All questions, B=3 | 83.3% | 81.7% | 51.3% |

| Method / budget | Maximum-effect discovery | Mean time relative to six-source replay |
|---|---:|---:|
| Recursive, B=1 | 7/12 | 0.703x |
| Source-ID, B=1 | 7/12 | 0.220x |
| Recursive, B=2 | 9/12 | 0.883x |
| Source-ID, B=2 | 10/12 | 0.378x |
| Recursive, B=3 | 11/12 | 1.030x |
| Source-ID, B=3 | 11/12 | 0.543x |

## Historical: Earliest source-local pilot: unstable small development samples, sum-log-p effects

Source and qualifications: [PILOT_REVIEW.md](v3/PILOT_REVIEW.md). These cohorts/targets differ from the main matrix; do not pool them as independent confirmation.

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

## Historical: Anchored training: adaptive development, not final confirmation

Source and qualifications: [ANCHORED_RESULTS.md](v4/ANCHORED_RESULTS.md). These cohorts/targets differ from the main matrix; do not pool them as independent confirmation.

| System | Mean QA F1 (%) | Singleton effect MAE | Normalized error |
|---|---:|---:|---:|
| Literal lexical baseline | 37.54 | 0.8475 | 1.377 |
| Anchored, factual-only | 42.61 | 0.8688 | 1.411 |
| Anchored, unpaired coefficient 0.1 | 43.10 | 0.8767 | 1.424 |
| Anchored, paired coefficient 0.1 (primary) | 43.18 | 0.8281 | 1.345 |
| Anchored, unpaired coefficient 1 | 42.40 | 0.8176 | 1.328 |
| Anchored, paired coefficient 1 | 41.94 | 0.8165 | 1.326 |

## Historical: Earlier FP32 frontier: Hotpot 96 / MuSiQue 32 per reader

Source and qualifications: [FP32_COMPLETE_RESULTS.md](v4/FP32_COMPLETE_RESULTS.md). These cohorts/targets differ from the main matrix; do not pool them as independent confirmation.

| Study | 128-to-1024 error reduction | MAE difference [95% interval] | Frozen-512 error reduction | MAE difference [95% interval] |
|---|---:|---|---:|---|
| hotpot_qwen7 | 61.3% | -0.7698 [-1.0847, -0.4816] | 16.1% | -0.1072 [-0.2198, 0.0049] |
| hotpot_llama8 | 55.6% | -0.3878 [-0.7015, -0.1136] | 22.0% | -0.0969 [-0.1608, -0.0457] |
| musique_qwen7 | -8.2% | 0.0368 [-0.2238, 0.3783] | 48.1% | -0.3198 [-0.7349, -0.0632] |
| musique_llama8 | 9.9% | -0.0312 [-0.2319, 0.1438] | 52.7% | -0.2314 [-0.5657, -0.0065] |

| Study | Raw QA F1 | Recomputed-512 QA F1 | NMAE 128 / 256 / 512 / 1024 | Frozen-512 NMAE |
|---|---:|---:|---|---:|
| hotpot_qwen7 | 58.93% | 57.87% | 2.106 / 1.762 / 1.115 / 0.814 | 0.935 |
| hotpot_llama8 | 71.00% | 64.57% | 1.718 / 1.355 / 1.084 / 0.763 | 0.845 |
| musique_qwen7 | 19.66% | 18.75% | 1.566 / 1.638 / 2.308 / 1.694 | 1.198 |
| musique_llama8 | 13.97% | 7.72% | 1.909 / 1.694 / 2.658 / 1.720 | 1.257 |

## Historical: Position/replacement ablation: numerical gate failed

Source and qualifications: [REPLACEMENT_LAYOUT_RESULTS.md](v4/REPLACEMENT_LAYOUT_RESULTS.md). These cohorts/targets differ from the main matrix; do not pool them as independent confirmation.

| Study | Position expansion with evidence fixed | Make replacement tokens visible | Actual replacement minus equal-length donor |
|---|---:|---:|---:|
| Hotpot / Qwen | +0.0025 [−0.0322, +0.0501] | +0.0991 [−0.0597, +0.3565] | +0.0774 [−0.0736, +0.3154] |
| Hotpot / Llama | +0.0074 [−0.0392, +0.0733] | +0.0451 [+0.0015, +0.0970] | −0.0191 [−0.0707, +0.0347] |
| MuSiQue / Qwen | −0.0034 [−0.0272, +0.0258] | +0.1652 [+0.0085, +0.4401] | +0.0893 [−0.0582, +0.3411] |
| MuSiQue / Llama | −0.0074 [−0.0311, +0.0094] | +0.2881 [−0.0267, +0.7377] | +0.2782 [−0.0551, +0.7411] |

Three of 1,440 deletions exceeded the preset equivalence tolerance. Equal-length donor intervals included zero throughout. These are diagnostic measurements, not a validated causal-mechanism result.

## Historical: Shared 400-question Hotpot secondary follow-up

Source and qualifications: [HELDOUT_FOLLOWUP_RESULTS.md](v4/HELDOUT_FOLLOWUP_RESULTS.md). These cohorts/targets differ from the main matrix; do not pool them as independent confirmation.

| Contrast | Qwen | Llama |
|---|---:|---:|
| BM25 refilling at fixed ranking | +0.1253 [+0.0771, +0.1790] | +0.0429 [+0.0197, +0.0696] |
| Additional BM25 ranking refit | +0.0596 [+0.0210, +0.1090] | +0.0411 [+0.0152, +0.0788] |
| Refilling minus ranking refit | +0.0658 [−0.0096, +0.1346] | +0.0018 [−0.0416, +0.0397] |
| RECOMP repacked minus frozen | +0.0739 [+0.0288, +0.1218] | +0.0284 [+0.0049, +0.0544] |
| BM25 gain minus RECOMP gain | +0.1110 [+0.0416, +0.1844] | +0.0556 [+0.0105, +0.1075] |

## Pending fresh-cohort results

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

No complete cohort is available in the November 25 snapshot. The primary and secondary criteria remain unevaluated. Do not populate these blanks using the older development budgets or the completed 512-token generalization results.

### Pending matched-length replacement controls (512 tokens)

| Reader | Fresh dataset | Selector | Actual-minus-donor raw-effect error | 95% CI |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | MuSiQue | BM25 |  |  |
| Qwen2.5-7B | MuSiQue | RECOMP |  |  |
| Qwen2.5-7B | 2Wiki | BM25 |  |  |
| Qwen2.5-7B | 2Wiki | RECOMP |  |  |
| Mistral7B-v0.3 | MuSiQue | BM25 |  |  |
| Mistral7B-v0.3 | MuSiQue | RECOMP |  |  |
| Mistral7B-v0.3 | 2Wiki | BM25 |  |  |
| Mistral7B-v0.3 | 2Wiki | RECOMP |  |  |

The control holds branch length and compact positions fixed while changing newly admitted content. Donor text is not guaranteed irrelevant. Neither these controls nor raw-deletion fidelity establish unique causal source credit.

## Provenance

Aggregate SHA256: `2f5ea9a3d131aa5600beb45aea27ca4f8166ed605ead158fac91d14e574f408a`.
New-run status comes from STABILITY_STATUS_20261125.json; no model scores were fit or selected to fill pending tables.
