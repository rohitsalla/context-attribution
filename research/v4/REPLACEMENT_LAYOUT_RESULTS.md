# Context length and replacement evidence after deletion

Both MuSiQue jobs, 8586479_2 and 8586479_3, completed with exit code zero, as did both Hotpot jobs and the CPU review. The independently reproduced review matches ARC exactly. Each model and benchmark uses 24 development questions. Questions are shared across the two models within each benchmark.

Positive differences below mean higher error against the raw-context source deletion effects. Intervals are unadjusted 95% question bootstrap intervals.

| Study | Position expansion with evidence fixed | Make replacement tokens visible | Actual replacement minus equal-length donor |
|---|---:|---:|---:|
| Hotpot / Qwen | +0.0025 [−0.0322, +0.0501] | +0.0991 [−0.0597, +0.3565] | +0.0774 [−0.0736, +0.3154] |
| Hotpot / Llama | +0.0074 [−0.0392, +0.0733] | +0.0451 [+0.0015, +0.0970] | −0.0191 [−0.0707, +0.0347] |
| MuSiQue / Qwen | −0.0034 [−0.0272, +0.0258] | +0.1652 [+0.0085, +0.4401] | +0.0893 [−0.0582, +0.3411] |
| MuSiQue / Llama | −0.0074 [−0.0311, +0.0094] | +0.2881 [−0.0267, +0.7377] | +0.2782 [−0.0551, +0.7411] |

All four position contrasts include zero. Making replacement tokens visible increases mean error in each study, but both studies whose activation intervals exclude zero also exceed the preset numerical tolerance in the separate compact-versus-dense check. Three of 1,440 source deletions exceed the 0.0001 limit, one for Hotpot/Llama (maximum 0.00010582) and two for MuSiQue/Qwen (maximum 0.00012207). All archived-score and sampled masked-token identity checks match exactly. The preset decision rule therefore withholds a mechanism conclusion. The equal-length donor contrasts also include zero in every study, and the result is conditional on the chosen donor text.

## Paper direction

Lead with the impact of the deletion protocol on audit fidelity. With factual evidence and answers fixed, deleting only the selected chunks from the removed source produces lower mean error than reranking and repacking on all four development studies. Three of those exploratory intervals exclude zero. The fixed-ranking comparison also favors deletion without refill in all four point estimates, with intervals excluding zero in two. These results describe agreement with the raw-context deletion reference for the tested pipelines.

The replacement study is a follow-up ablation. Its point estimates suggest that making extra text visible matters more than position expansion, while the identity of replacement text remains unresolved. A held-out comparison of deletion protocols against the raw-context reference is the direct next test of the proposed lead claim. Include absolute normalized error and the zero-effect baseline because the earlier MuSiQue experiments still exceeded that baseline even under frozen deletion.
