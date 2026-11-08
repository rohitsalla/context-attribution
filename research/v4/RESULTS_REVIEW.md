# Completed V3: no demonstrated paired-effect win

All nine learning runs, three 96-question control profiles, corrected canary evaluation and the 80-record mechanism diagnostic completed successfully. Program completion is not hypothesis success. Source hashes for all nine training manifests match the frozen public implementation. Compact results and trace hashes are in `V3_RESULTS.json`.

## Learning outcome

Three-seed means on the same 32 development questions:

| Objective | QA F1 (%) | Singleton effect MAE (mean log p) | Pooled normalized error |
|---|---:|---:|---:|
| Factual-only | 33.09 | 0.7613 | 1.210 |
| Unpaired perturbation distillation | 21.57 | 0.8595 | 1.366 |
| Paired-effect | 16.52 | 0.8442 | 1.341 |

Paired training is descriptively worse than factual-only by 16.57 F1 points and 10.9% effect MAE. It has no reliable advantage over unpaired training; the question-bootstrap effect-difference interval spans zero. All three systems have normalized error greater than the zero-effect predictor's value of one. These observations do not establish that paired supervision is useful.

**Matching failure:** initial parameter hashes agree within each seed, but all post-warm-up hashes differ. At step zero the forward loss agrees while gradient norms differ across supposedly identical runs; divergence is already visible at step one. Identical seeds were not enough. GPU backward nondeterminism is a plausible cause, but the exact operation has not been isolated. The nominal matched-objective contrast is therefore confounded; its intervals cannot be interpreted as a clean causal effect of changing the objective. Preserve these failed comparisons rather than silently replacing them. V4 branches from one checkpoint and optimizer state, forces deterministic math attention, and checks repeated same-state gradients.

## Expanded controls overturn the optimistic eight-case impression

QA F1 on 96 development questions, 512-token upper cap:

| Answerer | Raw | Local lexical | Local lead | Lexical pooled effect error |
|---|---:|---:|---:|---:|
| Qwen | 58.12 | 44.58 | 39.29 | 1.344 |
| Llama | 71.00 | 58.49 | 52.95 | 1.055 |
| Mistral | 64.26 | 50.05 | 45.22 | 1.494 |

The original eight examples were not representative of the full development mean. This is a reason to scale evidence before promoting a claim. All three lexical effect errors exceed the zero-effect baseline. These are fixed equal-per-source extraction implementations, not an impossibility result about compression. Stronger global sentence selection and oracle evidence controls are needed to identify whether source quotas, evidence loss, or reference behavior explain the result.

## Mechanism diagnostic

On eight original examples, question-aware global compression preserves high factual QA while substantially changing source effects; actual deletion-branch answers also differ in some cases. This rules out treating every observed difference as a citation-parsing artifact. It does not establish its population prevalence.

The alias stress test reduces raw QA F1 from 87.5% to 37.5%. The transformation therefore changes the task's difficulty and cannot cleanly identify a compressor knowledge-origin mechanism. Question-blind compression also worsens task quality. These findings do not support a clean claim that parametric knowledge is the main cause. Keep the missing-relation reconstruction witness as motivation, not as a proved mechanism or novel discovery.

## Strongest defensible direction

Study the distinction between evidence-grounded compression and preservation of a model's executed intervention response. Test stronger conventional extraction, gold-annotation evidence ceilings and a second natural benchmark. Support annotations are not causal truth: passages outside them may contain partial or duplicate evidence. Preserving a raw model's influence also preserves its distractor sensitivities and possible mistakes; this may differ from improving evidence quality.

Keep learning as a bounded, falsifiable branch: repair the exact-state control and test a stable lexical identity with learned residual edits. Do not scale pure soft memory, claim paired-effect novelty, or run the locked test before these gates pass. Counterfactual/causal distillation has substantial prior art, including [Causal Distillation for Language Models](https://arxiv.org/abs/2112.02505), [attribution preservation under network compression](https://arxiv.org/abs/2010.15054), and the CoD/PAIR/CICL work in the V3 protocol. The empirical distinction and any repair must earn their contribution through strong controls.
