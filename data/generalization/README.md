# Generalization cohorts frozen November 20, 2026

MuSiQue supplies 240 answerable development questions, 80 per two-, three- and four-hop stratum. The pinned mirror is [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) at revision `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321`. The [official MuSiQue repository](https://github.com/StonyBrookNLP/musique) distributes the dataset under CC BY 4.0 and notes possible single-hop overlap with source datasets including Natural Questions.

2WikiMultiHopQA supplies 200 validation questions, 50 per question type, from the [framolfese/2WikiMultihopQA mirror](https://huggingface.co/datasets/framolfese/2WikiMultihopQA) at revision `fe713bfbd1afbca1a65246741a75890405d56a3a`. The [official 2Wiki repository](https://github.com/Alab-NII/2wikimultihop) and mirror identify Apache 2.0 as the dataset license.

Rows are selected by a fixed question-ID hash within the strata, with normalized-question duplicates removed. The pinned Hotpot train, development and 400-question evaluation cohorts, plus the previous 96-question MuSiQue cohort, are excluded before selection. The preparation retains questions, answers, aliases where provided, and source paragraphs. Each paragraph receives an index-based source ID. Supporting facts and reasoning annotations are excluded from model inputs. Model scores are never used to select the cohort.

Raw and output hashes are recorded in `manifest.json`. Byte equivalence to the official original download has not been independently checked. The question disjointness check does not measure overlap of underlying Wikipedia paragraphs, single-hop source questions, selector training data or model pretraining. Preserve the original attribution and licenses when redistributing these derived records.
