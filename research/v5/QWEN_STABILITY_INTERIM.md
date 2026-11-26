# Qwen stability study: interim results

Verified 26 November 2026, 12:04 p.m. Eastern. Both Qwen cohorts are complete: 84 development / 240 test MuSiQue questions and 80 development / 240 test 2Wiki questions. Mistral remains pending; the prespecified two-reader primary criteria cannot yet be assessed.

These are individual-reader secondary analyses. Each policy receives a separate nonnegative effect scale fitted exclusively on development questions. This is effect rescaling, not probability calibration. Intervals use the frozen stratified question bootstrap (5,000 draws), condition on fitted scales, and are unadjusted for multiple comparisons.

## Primary budget: 512 tokens

| Dataset | Selector | Unscaled error reduction | Rescaled error reduction | Rescaled frozen minus recomputed MAE (95% CI) |
| --- | --- | ---: | ---: | --- |
| musique | BM25 | 23.89% | -0.29% | 0.00087 [-0.00575, 0.00795] |
| musique | RECOMP | 21.20% | 2.02% | -0.00596 [-0.00962, -0.00235] |
| 2wiki | BM25 | 16.01% | 11.82% | -0.05453 [-0.09539, -0.01307] |
| 2wiki | RECOMP | 14.33% | 9.36% | -0.04479 [-0.07226, -0.01711] |

Positive relative error reduction favors frozen deletion. Negative paired MAE differences favor frozen deletion. BM25 recomputes rankings and packing; RECOMP repacks its retained ranking.

At 512 tokens, both 2Wiki comparisons favor frozen deletion after rescaling, with intervals excluding zero. MuSiQue BM25 is unresolved; MuSiQue RECOMP retains a small advantage. Frozen deletion also beats the zero-effect baseline in the latter three comparisons, while MuSiQue BM25 does not establish an advantage.

## Secondary budget analysis

| Dataset | Budget | BM25 rescaled reduction | RECOMP rescaled reduction |
| --- | ---: | ---: | ---: |
| musique | 256 | -4.27% | -9.15% † |
| musique | 512 | -0.29% | 2.02% * |
| musique | 1024 | 0.83% * | -3.15% |
| musique | 2048 | 1.24% | 7.53% * |
| 2wiki | 256 | -1.38% | 3.82% * |
| 2wiki | 512 | 11.82% * | 9.36% * |
| 2wiki | 1024 | 8.86% | 8.52% * |
| 2wiki | 2048 | 1.26% | 2.49% * |

* Paired 95% interval favors frozen deletion. † Paired interval favors recomputation. Unmarked intervals include zero.

## Interpretation

The evidence supports a dataset- and budget-dependent advantage rather than a universal gain. Rescaling substantially reduces the original MuSiQue advantage, especially for BM25. On 2Wiki, meaningful residual improvements remain at the 512-token primary budget. Larger budgets do not produce a monotonic relative advantage. The paper should distinguish effect magnitude mismatch from residual selection instability, and assess absolute fidelity against zero. Mistral replication and the prespecified two-reader aggregate remain necessary before confirming the main claim.

The existing Mistral chain remains unchanged. Its first shard was pending on scheduler priority, with an estimated start of 26 November at 17:50 Eastern; this estimate is not guaranteed. No further GPU study was submitted for this interim analysis.

Source: immutable ARC study `context-attribution-stability-b74241d805ce`; frozen review plus read-only single-reader interval analysis. Full values and intervals: [QWEN_STABILITY_INTERIM.json](QWEN_STABILITY_INTERIM.json).
