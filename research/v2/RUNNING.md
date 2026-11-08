# Running V2

Install the model extra (`pip install -e '.[model]'`). Rebuilding the frozen dataset additionally needs `pyarrow`; model runs consume the included JSONL and do not need it. Dataset terms and transformations are in `data/v2-hotpot/README.md`. Model weights are not distributed here; follow each model card's terms and access requirements.

Create a private JSON mapping the four keys in `models.json` (`qwen15`, `qwen7`, `llama8`, `mistral7`) to local snapshot directories. Store it under ignored `logs/`. Each directory must contain the pinned weights and tokenizer/config files; the launcher checks shard presence before loading. It never downloads on an allocated GPU. A path to a different revision is not interchangeable: verify downloaded snapshot hashes when creating this site-local registry.

```sh
export PYTHONPATH="$PWD/src"
python -m context_attribution.v2.launch --paths logs/model_paths.json --check-only
python -m context_attribution.v2.launch --paths logs/model_paths.json --preflight
```

On ARC, export `PYTHON_BIN` (the installed model environment), `PROJECT_ROOT`, and `MODEL_PATHS` (absolute private registry path). Submit `run_arc_v2.slurm` using your account, GPU partition and QoS. Run a short `PREFLIGHT=1` allocation first; after success submit an array `0-5%1`, keeping at most one GPU active. Tasks 0–3 are the four hard-memory profiles; tasks 4–5 are matched factual-only/paired-effect training canaries. The default 55-minute task limit bounds total array allocation to 5.5 GPU-hours; interrupted hard-memory shards can resume only with matching source/data/config fingerprints. Training canaries require fresh output directories.

```sh
python -m context_attribution.v2.analyze --root results/v2-dev
```

The summary explicitly reports incomplete profiles. Eight development questions are a feasibility check, not a confidence interval or a held-out paper result. The feasibility runner rejects locked-test examples. Cold validation and common-raw-answer diagnostics are charged separately from production answer-plus-audit timing. No-context outputs are diagnostic and excluded from raw audit costs. Preserve the source commit, dataset manifest, environment package versions, Slurm logs, model revisions and raw JSONL before interpreting any result.
