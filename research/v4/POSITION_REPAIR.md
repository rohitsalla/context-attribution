# Failed positional probe and explicit-mask repair

The first positional probe completed all 96 questions but failed validation: the largest dense-versus-sparse mean-log-probability discrepancy was 32.204. Its apparent positional effects are invalid as a content/layout comparison. Keep `results/position-probe` and its original source commit `04a7e48` intact.

Inspection of installed Transformers 5.17.0, `masking_utils._preprocess_mask_arguments` and `find_packed_sequence_indices`, identified the cause: with no attention mask or past cache, nonconsecutive position IDs trigger packed-sequence detection. The old sparse input therefore split attention connectivity at the gap. This was an implementation error, not evidence that retaining positions creates a large influence change. Existing ordinary compact-position deletion scores agree exactly with the prior experiment.

Revision 2 always supplies an explicit 2D attention mask, including all-ones masks for sparse single-stream inputs. A regression check requires that gapped position IDs never omit this mask. Dense/sparse equivalence is tested within the same forced math-attention backend; default-versus-math numerical drift is separately reported. The primary position comparison uses ordinary SDPA, the same surviving tokens and input shape, and different declared position IDs. Do not interpret output if equivalence controls fail.

A one-question smoke must pass before the complete 96-question rerun, within one four-minute allocation. Outputs use new folders, preserving the invalid run. The model/data/target tokens and tolerance remain unchanged. This repairs attention semantics; it is not an evidence-based change to the positional hypothesis.

## Resource reallocation and stronger budget control

The matched-allowance global extractor reaches 56.60 F1, materially above the anchored model's approximately 43 F1, but with different effect distortion. Compare its smaller allowance against cap 512 on the **same** math-attention backend, fixed teacher targets and hash seed. The older 512 study used a different backend/reference; it is not sufficient for isolating the budget effect.

Defer the still-pending single-seed pure-soft audit and reallocate its 25-minute limit to the four-minute positional repair and a 15-minute same-runtime 512 control. All fifteen anchored branches across all three seeds remain completed and reported. This scheduling/scope decision does not hide an unfavourable completed run; the pure audit has consumed no GPU and its separate question remains unresolved. The resulting maximum stage allocation is 5 hours 49 minutes, with one GPU active. The cross-model/MuSiQue studies remain queued, and the final CPU report follows them.
