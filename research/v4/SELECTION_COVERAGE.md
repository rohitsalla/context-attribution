# Completed source-ownership audit of prior BF16 studies

CPU job 8553168 completed in one minute, exit zero, without using a GPU. All 576 factual global-extraction memories were reconstructed exactly using their pinned tokenizer and the provenance-aware selector; decoded memory text and achieved lengths match every stored trace. The source is commit `ef86536`; `review_selection_coverage.py` reproduces the result.

These are **previous BF16 measurements**, not outcomes of the pending FP32 study. Influence mass is the sum of absolute common-target mean-log-probability deletion responses. A fraction of mass is not a fraction of questions or a proportion of causal explanations.

| Study | Sources without selected tokens / all deletions | Raw influence mass on those sources | Compressed influence mass on those sources |
|---|---:|---:|---:|
| hotpot_qwen7 | 227 / 960 | 5.88% | 5.52% |
| musique_qwen7 | 1054 / 1920 | 35.31% | 14.10% |
| hotpot_llama8 | 207 / 960 | 12.80% | 3.70% |
| musique_llama8 | 1029 / 1920 | 39.59% | 11.79% |
| hotpot_mistral7 | 299 / 960 | 11.53% | 5.66% |
| musique_mistral7 | 1153 / 1920 | 37.03% | 16.68% |

On MuSiQue, unrepresented sources account for 35–40% of raw influence magnitude, while 12–17% of the recomputed pipeline's own influence magnitude is attached to those sources. This motivates testing both content omission and selection-policy effects. It does not prove that freezing selection improves fidelity or that selection alone causes these BF16 effects. Numerical sensitivity, changed sequence lengths and changed replacement content still matter.

For a frozen selector, removal of a source owning no factual chunks leaves the input identical and has zero effect by construction. Its raw-influence mass therefore provides a simple error lower bound for that declared frozen contract; this elementary fact is not a novel theorem. The pending FP32 study tests the frozen/recomputed contrast with identical factual tokens, explicit source ownership and independently checked compact-position scoring.

The original primary FP32 comparisons remain unchanged. The additional policy-difference analysis is exploratory, specified before the pending FP32 outcomes but motivated by these already observed BF16 traces. See `SELECTION_HYPOTHESIS.md` for the candidate, falsification boundary and primary related work.
