# Follow-up to the held-out deletion result

The primary held-out BM25 result is already confirmed. This offline follow-up asks where its MAE reduction arises. It classifies each deleted source by whether it owned any tokens in the factual compressed context. For each question it sums the signed error improvements in each group and divides by the total number of sources, preserving the original equal question weighting. The two contributions add to the original mean improvement.

| Held-out reader | Gain from represented sources | Gain from unrepresented sources | Represented share of total gain |
|---|---:|---:|---:|
| Qwen | 0.1597 | 0.0252 | 86.4% |
| Llama | 0.0619 | 0.0221 | 73.7% |

Most of the measured gain occurs on deletions of sources already represented in the compressed input. These additive error contributions describe where the gain occurs. They do not isolate the mechanism behind those changes.

On the 96 shared development questions, the paired BM25-minus-RECOMP difference in the frozen-deletion gain is +0.0635 for Qwen (95% interval −0.0724 to +0.1980) and +0.0592 for Llama (−0.0309 to +0.1481). Both readers use identical raw answer tokens across the selector runs. Raw effect scores differ by up to 0.00037 across runtimes, so this is an exploratory paired comparison of existing runs. Reproduce with `PYTHONPATH=src python3 review_selection_followup.py`. The confidence intervals do not establish that BM25 and RECOMP differ in how much frozen deletion helps.

The next experiment should add fixed-ranking repacking to the existing held-out BM25 comparison, concentrating on the primary all-source difference and separately reporting the contribution from represented sources. Keep the original BM25 candidate order, remove candidates from the deleted source and repack to the same cap. Compare that arm with frozen deletion to measure the impact of repacking under fixed scores. Compare it with full recomputation to measure the added effect of refitting the ranker. Use the same 400 questions, reader and fixed raw target as the archived comparison. Reuse the archived controls after factual and repacked score validation. Any follow-up on these already scored questions is a secondary analysis.

For transfer beyond BM25, run RECOMP on that same evaluation cohort with the same raw targets and scoring environment. Prespecify the selector-by-deletion-policy contrast directly. This can distinguish a smaller learned-selector benefit from a development comparison with wide intervals. Preserve the confirmed BM25 result and report the follow-up as a separate test.
