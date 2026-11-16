# Refilling, ranking refits and RECOMP on the same Hotpot cohort

This secondary analysis uses the same 400 previously scored Hotpot questions per reader and archived raw greedy answer tokens. It adds one BM25 arm and two RECOMP arms. It preserves the original held-out confirmation and interprets the new analysis in light of the previously observed results.

## Comparison

At the same 512 reader-token cap, freeze the factual BM25 sentence priority, exclude the deleted source, and greedily repack. Compare with the archived frozen and fully refitted arms. The path through fixed-ranking repacking decomposes the total error difference into a repacking contrast and a ranking-refit contrast. Tokens displaced by packing and changes in the resulting context length are part of the repacking contrast. This comparison concerns those operational choices under the tested pipeline.

Run the pinned Natural Questions extractive RECOMP scorer and the same project packing adapter as the development comparison. Its sentence scores depend on the query and sentence, so recomputation filters the surviving ranking and runs packing again. Both RECOMP arms start from identical factual tokens. The scoring reference is the same raw deletion response vector used by BM25 for each reader and question.

Use archived FP32 math attention, explicit causal masks and fixed answer spans. Verify the trace hash against the original result receipts and reconstruct the exact BM25 factual token hash. On each question, rescore the raw and compressed factual prompt and one ID-selected raw, frozen and fully recomputed deletion. Each must reproduce the archived score or effect within 0.0001 mean log probability. This tolerance applies to matched scores across A100 jobs. Abort and investigate any violation. The smoke uses two previously evaluated development questions per reader and checks the real checkpoint and all arms.

## Decision rule

The primary unit is the question. Give every question equal weight after averaging absolute errors over its sources. Use 5,000 paired bootstrap samples at seed 20261109 on sorted question IDs. Report the two BM25 path differences and their difference, all in MAE units. A positive difference indicates increased error. Favor refilling as the larger contributor only if its own interval and its difference from reranking are positive on both readers. Apply the symmetric rule for ranking refits. Otherwise report the model-specific contrasts and keep the dominant contribution unresolved. Report represented-source contributions as secondary diagnostics with the all-source denominator.

Report the RECOMP repacked-minus-frozen MAE difference and the paired difference between BM25 and RECOMP gains. The finding extends to this learned selector on both readers if both RECOMP intervals are entirely positive. A difference between selector gains is a separate contrast. Report absolute MAE and pooled NMAE against the zero-effect predictor. These intervals are exploratory and unadjusted. The two readers share the same questions.

Eight deterministic 50-question shards per reader reuse the original shard membership. A development smoke gates the serial GPU chain. A CPU review waits for all shards, validates complete coverage and produces the direction under the fixed rules.
