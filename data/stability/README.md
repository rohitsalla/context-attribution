# Fresh stability-study cohorts, frozen November 24, 2026

These derived records come from the same revision-pinned public source mirrors as [the generalization cohorts](../generalization/README.md). Credit the original [MuSiQue project](https://github.com/StonyBrookNLP/musique), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and [2WikiMultiHopQA project](https://github.com/Alab-NII/2wikimultihop), distributed under Apache 2.0. Preserve original attribution and license conditions when redistributing derived records. Source mirrors, raw hashes and revisions are recorded in manifest.json.

The changes from source data are deterministic question selection, normalization-based deduplication, split assignment, paragraph-to-source conversion, and extraction of support IDs for evaluation. Model inputs include the question and source text, never gold answers or support annotations. There are 324 MuSiQue questions (84 development, 240 test) and 320 2Wiki questions (80 development, 240 test). Each stratum is equally represented within each split. Every prior question under the existing project data directory is excluded before selection, including the preceding generalization cohorts.

This is question disjointness, not a document, pretraining, selector-training or semantic-near-duplicate contamination audit. Official-release byte equivalence of the pinned third-party mirrors is unverified. MuSiQue source material can overlap Natural Questions, which trained the selected RECOMP checkpoint. The fixed donor map pairs different questions within the same benchmark/split; donor content is not guaranteed irrelevant or inert.

See [the prospective protocol](../../research/v5/STABILITY_PROTOCOL.md). No reader scores were used for cohort selection.
