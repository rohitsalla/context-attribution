# Four-reader generalization results

All 100 GPU jobs (four six-question smokes and 96 full shards) completed successfully, covering 3,360 reader-question evaluations across four readers and three datasets. The final CPU review initially failed because its launcher omitted PYTHONPATH. Repaired review job 8951475 completed exit 0 in 10 seconds; the scientific runner and GPU outputs were unchanged. A standalone review launcher now sets the import path explicitly. Scientific source: b11f39e; protocol: GENERALIZATION_PROTOCOL.md.

## Main result

The prespecified generalization criterion is met: each of six dataset/selector contrasts has a negative upper paired question-bootstrap 95% confidence limit after averaging errors over the four fixed readers. This establishes a result for these readers, datasets and interventions, rather than every possible model or compressor.

| Dataset | BM25 error reduction, four-reader average | RECOMP error reduction, four-reader average |
|---|---:|---:|
| hotpot400 | 25.9% | 13.3% |
| musique240 | 26.8% | 13.0% |
| 2wiki200 | 20.8% | 10.6% |

Reductions above use the ratio of four-reader average MAEs, not the arithmetic mean of percentage reductions. Both selectors compare frozen deletion with their recomputed alternative; BM25 recomputes ranking and packing, and RECOMP repacks independently scored sentences.

## Individual BM25 replication

| Dataset | Qwen7 | Llama8 | Mistral7 | Qwen14 |
|---|---:|---:|---:|---:|
| hotpot400 | 24.7% | 21.2% | 24.4% | 29.5% |
| musique240 | 26.3% | 17.0% | 33.7% | 28.4% |
| 2wiki200 | 13.4% | 11.1% | 21.2% | 27.7% |

All 12 BM25 individual-model intervals favor frozen deletion. RECOMP individual-model intervals favor it on every Hotpot and MuSiQue cell, and on Mistral 2Wiki. Qwen7, Llama8 and Qwen14 RECOMP 2Wiki estimates favor frozen deletion but their intervals include zero. Individual-cell intervals are exploratory and unadjusted.

## Paired confidence intervals for the primary contrasts

| Dataset | Selector | Mean frozen-minus-recomputed MAE | Question-bootstrap 95% CI |
|---|---|---:|---|
| hotpot400 | BM25 | -0.16414 | [-0.23075, -0.11161] |
| hotpot400 | RECOMP | -0.07671 | [-0.10738, -0.04989] |
| musique240 | BM25 | -0.10310 | [-0.14500, -0.06938] |
| musique240 | RECOMP | -0.04182 | [-0.05650, -0.02823] |
| 2wiki200 | BM25 | -0.11329 | [-0.17395, -0.06059] |
| 2wiki200 | RECOMP | -0.05068 | [-0.09544, -0.01369] |

## Interpretation and boundaries

The supported paper claim is that, at the tested 512-token budget, freezing selected evidence improves fidelity to raw-context source-deletion effects compared with recomputing compressed context. Both a lexical selector and a learned extractive selector show this benefit after averaging over the four fixed readers. It is a controlled replication across three model families and a within-Qwen size comparison.

The primary analysis clusters model repetitions by question: each bootstrap resamples questions after averaging within-question differences over the four readers. It does not treat the 3,360 reader-question evaluations as independent questions, nor sample a population of model architectures. There are 840 question identities across the three datasets. Hotpot was used previously and remains a replication cohort; MuSiQue and 2Wiki were fixed before these model scores.

The result concerns attribution fidelity, not answer correctness, citation entailment or causal truth. Effects share the raw factual answer tokens. Keep absolute error, zero-effect baselines, compressed factual QA and severe-source decision diagnostics alongside the relative gains. Frozen deletion need not improve downstream answers. The mechanism cannot be summarized as refilling alone: fixed-ranking refilling and reranking have dataset-dependent contributions. Source mirrors, potential source exposure and contamination-audit limits remain as recorded in the protocol.

No GPU configuration or scientific scoring code was changed after reviewing the outcomes. The CPU launcher repair corrected imports only.
