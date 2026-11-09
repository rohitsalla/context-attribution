# Completed anchored comparisons: modest gains, not a paired-method win

All three anchored seeds completed, each with the same 96 development questions and five objective branches. Every branch's pool, optimizer, CPU RNG and CUDA RNG start hashes match its single shared warm-up. The initial lexical identity and repeated-gradient checks passed. No locked-test example was evaluated. `ANCHORED_REVIEW.json` and `review_anchored.py` contain all seeds, coefficients, trace hashes and paired question-bootstrap intervals.

| System | Mean QA F1 (%) | Singleton effect MAE | Normalized error |
|---|---:|---:|---:|
| Literal lexical baseline | 37.54 | 0.8475 | 1.377 |
| Anchored, factual-only | 42.61 | 0.8688 | 1.411 |
| Anchored, unpaired coefficient 0.1 | 43.10 | 0.8767 | 1.424 |
| Anchored, paired coefficient 0.1 (primary) | 43.18 | 0.8281 | 1.345 |
| Anchored, unpaired coefficient 1 | 42.40 | 0.8176 | 1.328 |
| Anchored, paired coefficient 1 | 41.94 | 0.8165 | 1.326 |

The primary paired loss reduces mean effect MAE by 4.7% versus factual-only. The paired question-bootstrap difference is -0.0407, interval [-0.0750, -0.0079]. Its QA difference is +0.57 points, interval [-1.79, +3.21]. This is a modest development signal, below the predeclared 20% improvement target.

Against the matched-coefficient unpaired control, the effect difference is -0.0486 with interval [-0.1384, +0.0098], spanning zero. Coefficient-one paired and unpaired results are essentially tied. The strongest unpaired setting has slightly lower error than the primary paired setting. There is no reliable special advantage for pairing itself. Joint-deletion transfer over factual-only is also inconclusive.

The primary anchor gains 5.64 F1 points over literal memory, but factual-only already gains 5.07 points. The additional QA gain from the paired objective is small and uncertain. The effect-error difference versus literal memory is likewise uncertain. All native singleton normalized errors remain above the zero-effect predictor's value of one. Ranking and downstream audit utility are separate secondary questions; do not switch the headline metric to rescue the method.

Intervals bootstrap questions after averaging seed differences; they do not estimate the full uncertainty across possible training seeds. These are adaptive development results, not confirmatory significance claims. Do not scale the learned method or declare an ICML contribution from this evidence.

## Stronger extraction and the budget frontier

Global extraction under the anchor's same factual allowance achieves 56.60 F1 with effect MAE 1.0024. It answers much better, while the anchor has lower effect error. This is a tradeoff, not a blanket domination claim. Global extraction uses at most the same allowance and may leave space unused; it can refill after source deletion, unlike fixed local slots.

On the **same runtime and exact teacher targets**, expanding global extraction to cap 512 raises QA F1 from 56.60 to 57.87 (point difference +1.27 points; interval [-5.40, +7.95]) and reduces effect MAE by 30.3%, from 1.0024 to 0.6986 (difference -0.3038; interval [-0.5403, -0.0837]). Joint error also falls. Actual average memory grows from 353.4 to 506.2 positions. The larger system's error still exceeds the zero-effect baseline.

For the 75 questions with unchanged normalized factual answers, the effect-error difference interval includes zero. Thus this cohort does not statistically establish an independent budget effect while answer identity is held fixed. The promising direction is a measured quality/influence/resource frontier, replicated across the already scheduled models and MuSiQue—not a claim that QA and influence are universally independent. See `BUDGET_COMPARISON.json`.

## Positional control: not validated yet

The first positional probe accidentally triggered packed-sequence masking and is invalid. The explicit-mask repair removed that implementation issue, but dense/sparse numerical equivalence still fails on some cases: maximum discrepancy 0.746 mean log probability, and maximum default-versus-math backend discrepancy 1.452. The ordinary deletion scores reproduce exactly. The alternate retained-position aggregate must not be interpreted as a reliable positional mechanism until the failure is resolved.

A selected-case FP32/explicit-4D-mask diagnostic will distinguish attention-mask construction from arithmetic precision. The failures remain preserved. Runtime-conditioned deletion experiments remain records of the executed implementation; interpreting them as semantic evidence influence requires these additional controls. Numerical/backend sensitivity is not a new discovery claim, nor an excuse to discard unfavorable results.

The selected worst failure now passes the full-model CPU FP32 check (maximum dense/sparse discrepancy 0.0000513; 2D/4D discrepancy zero). This supports corrected semantics for that case, with the hardware/precision scope explicitly limited. The selected-case CUDA FP32 diagnostic remains queued; this CPU pass does not validate the full BF16 positional aggregate.

## Next decision

Do not make paired-effect training the paper's headline. Complete cross-model and second-benchmark controls, validate intervention semantics, and test whether effect gains exceed ordinary perturbation supervision, token allocation and confidence calibration. A temperature-only control is also needed before attributing small probability-effect gains to learned evidence preservation: positive rescaling can alter probabilities while leaving greedy decisions unchanged. That control is outstanding, not an executed result.
