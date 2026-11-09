# Selected-case precision and mask diagnostic

Designed after revision-2 positional equivalence failed. This is post-hoc debugging on deliberately selected failures, not a representative sample or a new positional-effect estimate.

Select the union of six largest dense/sparse errors, six largest default-versus-math errors, and four smallest dense/sparse errors from the completed 96-question probe. Retain the same two question-ID-selected source deletions, target tokens and model weights. Record all selected IDs and the input trace hash before inference.

For each case, compare sparse retained-position removal with dense masking using (a) explicit 2D masks processed by Transformers and (b) directly constructed 4D causal/key masks that bypass mask inference. Force math attention for all comparisons. First run the native BF16 model, then upcast the same parameters to FP32 with TF32 disabled and float32 matmul precision set to highest. The weights are not retrained. Score accumulation uses float64 log-softmax in both cases. Report every dense/sparse and 2D/4D difference, not just passing examples.

Also record the compact-versus-retained-position score gap on the same surviving token stream under explicit 4D masks. It is a diagnostic on selected cases only; do not extrapolate its magnitude or source-support fractions to all questions. If FP32 equivalence still fails, investigate semantics rather than certify the positional experiment. Upcasting changes runtime arithmetic and is not a claim of bitwise reproduction of BF16 results.

One standalone ten-minute GPU job follows the active benchmark task, then remaining benchmark tasks resume. Source and outputs are separate from the frozen V4 runtime and both earlier positional attempts. This brings the maximum stage allocation to 5 hours 59 minutes, with at most one GPU. The pending pure-soft audit remains deferred. No new learning grid or locked-test evaluation is authorized by this diagnostic.
