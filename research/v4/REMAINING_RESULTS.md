# Final review of the remaining V4 runs

All three stronger-compressor jobs, the selected CUDA precision diagnostic and CPU aggregation completed with exit code zero. The conditional full positional job also exited zero **because its numerical gate skipped execution**. A successful scheduler exit is not evidence that the full positional study ran. Raw records remain under ignored `results/`; `review_remaining.py` reproduces the public summary and validates completion, target alignment and hashes.

## Positional controls

The explicit 2D and direct 4D masks produce identical scores in all 30 selected source checks. BF16 dense/sparse-equivalent constructions still differ by as much as 0.745967 mean log probability. Upcasting the same weights to FP32 reduces the maximum discrepancy to 0.000163390, median 0.000001166. One of 30 checks fails the fixed 0.0001 gate. The selected set contains 15 questions chosen from previous failures and low-error controls; it is not a representative cohort.

Job 8551169 correctly wrote a skip record. Keep the preset tolerance unchanged, preserve all failures, and do not headline the previous 96-question retained-position aggregate as a validated mechanism. The CPU single-case pass and large FP32 error reduction are useful implementation diagnostics, not certification of the full GPU intervention study. This direction is paused while the budget/selection study proceeds with ordinary compact positions.

## Stronger generated-memory controls

Qwen2.5-7B compresses the same 32 Hotpot development questions for each answerer at nominal cap 512. NMAE below uses common raw-answer **mean-token** log probabilities; the JSON also preserves the older summed-token weighting explicitly. Zero-effect prediction has NMAE 1.

| Answerer | Raw QA F1 | Generated QA F1 | Actual memory positions | Effect NMAE |
|---|---:|---:|---:|---:|
| qwen7 | 69.38% | 64.29% | 45.41 | 2.566 |
| llama8 | 82.03% | 65.59% | 44.12 | 2.850 |
| mistral7 | 63.56% | 64.62% | 50.16 | 3.022 |

All 192 sampled cached/cold checks have identical tokens and zero effect difference. These checks certify sampled replay within the generated-memory pipeline; they do not certify agreement with raw-context effects. Each model's generated-minus-raw QA bootstrap interval includes zero. Do not infer reliable QA gains or losses from this 32-question pilot.

The nominal cap is strongly underfilled: achieved lengths are roughly 9–10% of 512. This is not an equal-achieved-length comparison with extraction or anchored memories. A larger compressor alone does not establish faithful transfer, but this result cannot show that generative compression as a class fails. Future broad comparisons need released methods and matched achieved resource curves.

## Continuing direction

Run the preregistered-in-repository exploratory `FRONTIER_PROTOCOL.md`: FP32 answering/scoring, caps 128/256/512/1024, two models and two benchmarks, plus a frozen-selection 512 branch with identical factual memory to its recomputed counterpart. Report raw passthrough and repeated-score checks, all budgets, answer quality, effect/rank recovery, and uncertainty. This tests budget and selection sensitivity; it does not repair or establish the positional mechanism. No further paired-training grid or locked-test evaluation is scheduled.
