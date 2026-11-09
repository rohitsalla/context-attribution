# Completed MuSiQue development results

All three 96-question studies completed. QA F1 is a fraction; NMAE compares singleton source effects on the same raw-generated target tokens. Zero-effect prediction has NMAE 1. These are adaptive development results, with no locked-test evaluation.

| Model | Raw F1 | Global-512 F1 / NMAE | Support oracle F1 / NMAE |
|---|---:|---:|---:|
| Qwen7 | 0.2062 | 0.1610 / 2.2832 | 0.5038 / 1.7627 |
| Llama8 | 0.1571 | 0.1497 / 1.7791 | 0.4588 / 1.8021 |
| Mistral7 | 0.1530 | 0.1114 / 1.9334 | 0.4039 / 1.0545 |

The annotation-informed oracle is an uncapped diagnostic ceiling, not a deployable compressor or causal ground truth. Its higher QA does not reliably recover the raw pipeline's probability-based source influence. All compressed NMAEs exceed the zero-effect reference here. Low raw QA (15–21%) also limits interpretation: preserve and report incorrect-answer slices rather than implying faithful knowledge attribution.

This supports measuring answering quality and audit transfer separately. It does not establish a universal conflict or a unique positional mechanism. The global budget effect observed on Hotpot still needs replication; this table contains one global budget, not a budget curve.

The 32-question Qwen7-compressor/Qwen7-answerer pilot also completed: raw F1 0.69375, global generated-summary F1 0.64286. This is a smaller cohort and a different protocol from the evidence table; no direct cross-cohort superiority claim is made. Other stronger-compressor families remain pending/running.
