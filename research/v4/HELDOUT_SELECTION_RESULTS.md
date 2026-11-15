# Held-out Hotpot and extractive RECOMP results

Frozen deletion passed the prespecified BM25 confirmation on both readers. All 16 GPU shards and the CPU review 8703535 finished. Local aggregation using the frozen source at `f94782d` verified all 400 questions per model and reproduced the ARC verdict, intervals and trace receipts. Aggregate floating-point roundoff agreed within 1e-12.

| Selector and split | Reader | Frozen MAE | Recomputed MAE | Frozen minus recomputed, paired 95% interval | Frozen / recomputed NMAE |
|---|---|---:|---:|---:|---:|
| BM25, held-out Hotpot (400) | Qwen | 0.5645 | 0.7494 | −0.1849 [−0.2496, −0.1264] | 0.907 / 1.205 |
| BM25, held-out Hotpot (400) | Llama | 0.3113 | 0.3952 | −0.0840 [−0.1316, −0.0473] | 0.851 / 1.080 |
| Extractive RECOMP, Hotpot dev (96) | Qwen | 0.6962 | 0.7399 | −0.0437 [−0.1396, +0.0391] | 1.168 / 1.241 |
| Extractive RECOMP, Hotpot dev (96) | Llama | 0.3743 | 0.4120 | −0.0377 [−0.1110, +0.0244] | 0.922 / 1.015 |

On held-out Hotpot, frozen deletion reduced source-effect MAE by 24.7% on Qwen and 21.2% on Llama at identical factual context and answers within each model. Frozen deletion fell below zero-effect prediction (NMAE 1) for both readers. Top-source recovery rose from 65.5% to 69.25% for Qwen and changed from 69.75% to 69.5% for Llama. The confirmed result concerns mean error in the magnitude of deletion effects.

RECOMP review 8703740 also completed and reproduced locally within 1e-12. The released Natural Questions scorer with the project's packing adapter gives favorable point estimates for frozen deletion on both readers. Both intervals include zero, so the prespecified development criterion for consistency across readers remains unmet. Qwen's frozen RECOMP NMAE remains above the zero-effect predictor. The RECOMP comparison uses development Hotpot rather than the locked questions, and the table does not estimate a selector-by-protocol interaction across splits.

Lead the paper with the held-out result for the tested BM25 selection pipeline and two readers. Report RECOMP as a scope check and keep the replacement and position experiments as exploratory. The evidence supports lower mean raw-reference deletion error for this BM25 pipeline and leaves generalization to independently scored learned selectors unresolved.
