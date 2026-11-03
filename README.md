# Replay-validated context attribution

Research code for tracing source influence through a summarize-then-answer pipeline. Based on Rohit Kumar Salla's November 2, 2026 project proposal. The implementation includes a reproducible pilot harness, complete source interventions, recursive local ranking, and budgeted replay evaluation. This is exploratory research infrastructure; no general attribution or efficiency guarantee is claimed. See [the pilot protocol](PILOT_PROTOCOL.md) for hypotheses and limitations.

## Run the infrastructure validation

```sh
git clone https://github.com/rohitsalla/context-attribution.git
cd context-attribution
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m context_attribution.experiment --backend fixture --output results/fixture
```

The fixture backend is a deliberately engineered symbolic system. Its probabilities and whitespace token counts are synthetic. It tests the experiment machinery, not model behavior. Results contain an explicit `research_evidence: false` marker. The 20 fictional questions have shuffled source order and eight development/twelve test labels. They share one linguistic template: family labels group entity variants and do not establish template generalization. This pilot is not the proposed 40/80 main split or final benchmark.

## Run a real model

Install the `model` extra, then choose a compatible 7–8B instruction model and pin its Hugging Face revision:

```sh
python -m pip install -e '.[model]'
python -m context_attribution.experiment \
  --backend hf --model MODEL_ID --revision COMMIT_SHA --device cuda \
  --limit 1 --caps 128 256 --output results/model-smoke
```

Replace the uppercase values. Begin with one example to validate chat templates, token alignment, generation limits, and hardware before running 20. Model loading may download weights. No paid service is used. The HF adapter disables reasoning when supported, uses greedy decoding by default, scores a fixed target token sequence without EOS, and sums its token log probabilities. Target tokens are encoded separately as an explicit continuation, consistently across interventions. Inspect generated answer spans before interpreting results; formatting failures or reasoning output can invalidate the intended short-answer study.

For an existing converted MLX checkpoint on Apple Silicon:

```sh
python -m pip install -e '.[mlx]'
python run_local_dev.py --model /path/to/converted-mlx-model
```

`run_local_dev.py` starts a bounded background development run and writes its PID and progress to `logs/`. Set `--python` to select a different runtime. The checkpoint is not included in this repository.

## Slurm execution and analysis

The launchers accept `PROJECT_ROOT`, `PYTHON_BIN`, `MODEL_PATH`, `MODEL_REVISION`, and `OUTPUT_DIR` through environment variables. Choose your own permitted account, partition, and QoS at submission time. Their default model is Qwen2.5-7B-Instruct at revision `a09a35458c702b33eeacc393d103063234e8bc28`.

```sh
mkdir -p logs
export PYTHON_BIN="$PWD/.venv/bin/python"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --qos=YOUR_QOS run_arc_fasttrack.slurm
```

The launchers use offline model loading by default; cache the pinned checkpoint first, or set `HF_HUB_OFFLINE=0` when downloads are allowed. The fast-track launcher requests one GPU for 30 minutes, one CPU core, and 32 GB of host memory. For serial continuations, submit the same launcher with `--dependency=afterany:JOB_ID --array=0-2%1` and the same output directory. Keep the source tree fixed while resuming: source and configuration mismatches are rejected.

Analyze completed or partially completed traces with:

```sh
python -m context_attribution.analyze results/qwen25-7b-arc
```

Generated datasets, traces, reports, model weights, and machine-specific logs are excluded from version control. They remain local to each experiment. No pretrained model weights are distributed here.

## Implemented

- Six-source deletion with complete summary regeneration and a fixed original answer span.
- Two unchanged-input replays and paired intervention repeats. Noise floor is the maximum observed unchanged-input shift, with a numerical minimum. This is a pilot diagnostic, not a calibrated statistical threshold.
- Raw-context answering diagnostic, exact-match accuracy, truncation flags, generated-answer traces, and support annotations.
- Two-stage sparse-surrogate local scoring: answer to summary sentences, then summary sentences to source blocks. Signed, absolute-L1-normalized coefficients are composed and ranked by magnitude. All summary sentences are used initially.
- Budgets 1/2/3, source-ID tracing, lexical overlap, full replay oracle, and exact expectation over all random subsets at each budget.
- Null effects for untested sources; absolute-effect discovery with noise ties; negligible-effect separation; ranking correlation and support recall in traces.
- Cluster bootstrap discovery intervals by base question, separately for development/test and all/original-correct slices.
- Per-call timing and token accounting including candidate scoring. Full-run cost includes offline reference construction; selected-method replay cost and selection overhead are separately recorded. These are accounting estimates from shared offline traces, not independent production latency measurements.
- Incremental JSONL traces, dataset hash, model revision, configuration, and reproducible seeds. Earlier output directories cannot be overwritten accidentally.

For the earlier-start ARC route, see `FASTTRACK_ROUTING.md` and `run_arc_fasttrack.slurm`. The job uses smaller 30-minute allocations with serial continuations. Matching completed conditions resume automatically inside Slurm; elsewhere supply `--resume`. Configuration, dataset, and source hashes must match, and changed configurations are rejected. Completed conditions are flushed and fsynced before the next condition starts.

The local scorer is a **ContextCite-style approximation**, not the official library or a validated reproduction. It follows random context masks, a response-probability logit transform, and an L1 linear surrogate. See the [official ContextCite repository](https://github.com/MadryLab/context-cite) and [paper](https://arxiv.org/abs/2409.00729). Citation IDs are retained in scored sentences, so their effect on ranking must be ablated before research conclusions. Lexical overlap is not a semantic-embedding baseline.

## Remaining research work

1. Complete the pinned 7B model pilot; verify fixed-span scoring independently and inspect summaries/answers. Analyze the smaller-model MLX smoke test separately, including its 8-bit quantization and hardware differences.
2. Add real-model irrelevant/misleading/duplicate stress-test runs; current versions of these controls are software tests only. Extend paired stochastic replication beyond two repeats and estimate uncertainty in intervention effects before interpreting them.
3. Compare the official ContextCite implementation, add semantic similarity, and freeze normalization, regularization, and sentence selection on development data.
4. Audit and freeze the 120-item dataset (80 fictional, 40 natural-text), preserving the proposed 40/80 family-disjoint split. No natural-text examples have been collected or audited yet.
5. Add paraphrase and consistent fact-replacement subsets, 512-token summaries, end-to-end latency measurements, paired method-difference confidence intervals, and failure analysis. The current CLI generates pilot fixtures only; external benchmark loading is still to implement.
6. Expand only after real-model effects exceed replay variation. A negative result is valid.
