# Completed study v1

All 548 evaluation conditions and 12 online-verification cases completed successfully, including the CPU finalizer. The inference snapshot is commit `40c5476bee6d1635fbb9fa7fbe2700651b9830ac`, using unmodified Qwen2.5-7B-Instruct on one H200 at a time. Dataset byte hashes, runtime source hashes, complete condition counts, fixed answer spans, and independent likelihood checks were verified after retrieval.

**The results show local attribution can misrank replay effects, but do not establish a useful efficiency advantage over cheap source-ID tracing.**

## Main discovery results

Discovery means finding a source whose absolute replay effect is maximal, allowing ties within replay noise. The three summary caps produced identical main-set summaries, answers, and effect vectors. The table therefore uses one cap (128); it does not treat the repeated caps as independent questions.

| Scope / replay budget | Recursive local ranking | Source-ID tracing | Random subset expectation |
|---|---:|---:|---:|
| Held-out test, B=1 (80 questions) | 40.0% | 33.8% | 17.7% |
| Held-out test, B=2 | 68.8% | 62.5% | 35.0% |
| Held-out test, B=3 | 78.8% | 73.8% | 51.9% |
| All questions, B=1 (120 questions) | 51.7% | 47.5% | 17.4% |
| All questions, B=2 | 74.2% | 73.3% | 34.4% |
| All questions, B=3 | 83.3% | 81.7% | 51.3% |

At B=1, the paired recursive-minus-source-ID difference is +6.25 percentage points on test, with a question-level bootstrap 95% interval of **[-1.25, +13.75] points**. Across all 120 questions the difference is +4.17 points, interval **[-0.83, +9.17]**. Neither excludes zero. The all-question comparison with random selection is stronger (+34.31 points; the original analysis interval is approximately +25.28 to +43.33 points), but random is not the strongest cheap baseline.

Original-answer accuracy was 71.25% on test and 76.67% overall; raw-context accuracy was 80% overall. Attribution influence is not evidence that an answer is correct. On the fictional subset, recursive B=1 discovery was 66.25%; on the facts-page subset it was 22.5%, versus 57.5% for lexical selection. Results do not transfer uniformly across these data types.

## Actual online execution

All twelve online cases matched the reference answer spans. Candidate scoring was included; each method truly replayed only its selected sources. Costs exclude the common original pipeline and compare with the saved six-source replay timing, so these remain small, hardware-specific measurements.

| Method / budget | Maximum-effect discovery | Mean time relative to six-source replay |
|---|---:|---:|
| Recursive, B=1 | 7/12 | 0.703x |
| Source-ID, B=1 | 7/12 | 0.220x |
| Recursive, B=2 | 9/12 | 0.883x |
| Source-ID, B=2 | 10/12 | 0.378x |
| Recursive, B=3 | 11/12 | 1.030x |
| Source-ID, B=3 | 11/12 | 0.543x |

At B=1, recursive selection achieved the same discovery as source-ID tracing at about **3.2x its time**. At B=3 it offered no wall-time saving over complete replay in this online subset. Across all 120 cases, the offline accounting estimate for recursive B=1 was about 0.67x complete-replay time but **4.70x the input-plus-output token count**. Wall time, model-call count, and token count must not be conflated.

## Controls and interpretation

- Summary length averaged 30.5 tokens; original summaries were never truncated at cap 128. Increasing caps to 256 and 512 changed no main outcomes. **Compression-budget sensitivity was not demonstrated.**
- Twelve duplicate-evidence cases showed much larger joint-removal effects than either individual deletion. Individual removal effects are not unique allocations of causal credit.
- Deliberately corrupted workplace records produced incorrect original answers in all twelve misleading-evidence cases at cap 128. Strong attribution can support an incorrect answer.
- One of eight stochastic conditions had unchanged-input replay variation larger than its largest estimated intervention effect; its source effects remain unresolved under the chosen diagnostic.
- The real-world subset consists of forty short official Nobel facts-page excerpts checked by the assistant. It is not an independently human-audited long-form natural-text benchmark. The scorer is a sparse-surrogate approximation, not an official ContextCite reproduction.

H1 has concrete conditional-pipeline counterexamples. H2 has not established an advantage over the strongest implemented cheap baseline. A publication-level novelty or efficiency claim is not justified by these results alone.

## Reproduction and evidence

`study_results.json` contains a public, path-free summary. Run `python summarize_completed_study.py results/full-main-v1` against the retrieved outputs to validate provenance and regenerate it. Raw traces and complete reports remain under the ignored local results directory; they were not uploaded with private runtime paths. The original generated JSON contains stale legacy wording about a single-template pilot; the actual main dataset has eight fictional surface-template families plus the test-only facts-page family, as specified in the frozen manifest and full-run protocol.
