# HotpotQA development and locked-test subsets

Source: [HotpotQA on Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa), revision `1908d6afbbead072334abe2965f91bd2709910ab`. Original benchmark: [Yang et al., HotpotQA, EMNLP 2018](https://arxiv.org/abs/1809.09600), with Wikipedia source text and its contributors. The HF dataset card identifies the dataset license as [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); these derived data retain that attribution and license. This notice applies to the dataset, not a new license grant for the project's code or model checkpoints.

Transformations: deterministic question-ID selection, stable paragraph IDs, source formatting, and explicit annotation-issue flags. Original question/answer/context text is retained. Duplicate titles are preserved. The source parquet byte hashes and selected JSONL hashes are in `manifest.json`. Raw downloaded parquet/HTML caches are excluded from Git.

There are 512 training questions, 96 development questions, and 400 locked questions drawn from the official validation split. Question IDs are disjoint; retrieval-document overlap is reported, not asserted absent. The locked questions have not been used for model evaluation in the v2 feasibility stage. Original benchmark annotations are not represented as a new independent human audit.
