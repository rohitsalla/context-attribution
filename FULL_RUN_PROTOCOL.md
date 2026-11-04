# Full study v1: fixed before main-dataset model outcomes

## Basis and interpretation

The legacy 7B pilot completed 40 conditions on an H200, with 20 questions at two caps. At budget one, recursive selection discovered the maximum-effect source in 85% of cases, versus 1/6 for random selection. Source-ID tracing also achieved 85%, at lower cost. Strict answer accuracy was 35%; many misses omitted the artificial numeric suffix in city names. These findings motivate a new, separately identified dataset; they do not establish a general advantage.

H1 and H2 remain those in `PILOT_PROTOCOL.md`. Expansion is justified by reproducible effects and concrete local-ranking misses. No positive publication finding is assumed.

## Frozen datasets and changes

`data/main-v1/manifest.json` records byte hashes. The main set has 120 base questions: 80 fictional questions from eight surface-template families, plus 40 official Nobel facts-page examples. Four fictional families (40 items) form development; the remaining four and all natural examples form test (80 items). Variants retain the base question's split. Main fictional city names contain no digits.

Each natural example has exact affiliation/location excerpts, official URLs, and cached-page hashes checked against official API fields. All forty name/institution/city rows were reviewed during preparation. `human_audited` is explicitly false. This is an assistant-checked **facts-page subset**, not a human-audited long-form natural-prose benchmark. The project's stronger original natural-text audit requirement remains a qualification, not a completed human audit. Numeric offsets refer to normalized page text, and omissions are marked `[...]`. Generic ambiguous institution names were excluded. Exact gold aliases are declared before inference, including the source's city/state rendering; unlisted model paraphrases remain incorrect.

The main local scorer retains 32 masks, alpha 0.01, signed absolute-L1 composition, and absolute ranking. The sentence parser now preserves initials and common abbreviations (e.g. St. Petersburg); this is a declared parser repair from the pilot. The run retains the same summarize/answer prompts, six blocks, and 24-token answer cap. No main-test outcome is used to choose parameters.

## Run matrix

1. Main: 120 questions x caps 128/256/512 = **360 conditions**. All six deletions, two unchanged-input repeats and two paired deletion repeats, raw-context diagnostics, and budgets 1/2/3.
2. Stress: twelve prespecified fictional bases x five variants (paraphrase, lexical distractor, duplicate, consistent fact replacement, misleading evidence) x three caps = **180 conditions**. Duplicate cases additionally remove the duplicated pair jointly. Variants share base IDs for uncertainty calculations.
3. Stochastic: eight prespecified fictional bases at cap 128, temperature 0.7, and five paired repeats = **8 conditions**. These are exploratory stochastic controls, not a calibrated universal noise threshold.
4. Actual online verification: twelve prespecified bases (eight fictional, four natural), cap 128, four selectors (recursive, lexical, source-ID, random), budgets 1/2/3. Each selector truly regenerates only its selected branches, charges candidate cost, and leaves all other effects null. It never reads full-reference scores to select sources. Offline comparisons require matching original answer spans.

Primary model: unmodified Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, BF16, no fine-tuning. Source hashes and dataset hashes protect resume. Masked-label cross-entropy independently checks explicit answer-token indexing at startup.

## Execution and completion

Use a separate checkout and results directory from the legacy pilot. The GPU array permits **one concurrent GPU**. Twelve 30-minute tasks provide two serial passes over six components; completed components skip inference and incomplete ones resume. Maximum GPU reservation is six hours; expected computation is lower. A dependent CPU finalizer checks all counts and writes aggregate analysis. It fails explicitly if any required conditions are missing.

Actual summary lengths, truncation, errors and incorrect answers remain visible. Caps may be nonbinding. Report all cases and original-correct slices, separate fictional and facts-page behavior, use base-question clusters for variants, include candidate overhead, and compare against source-ID tracing. Facts-page results cannot be presented as the completed original human audit or long-form benchmark.

The official ContextCite reproduction, an embedding-similarity baseline, optional attention scorers and smaller-model replication are not implemented by this run. Claims must remain scoped to the implemented surrogate and baselines.
