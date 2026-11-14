#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--root "$PROJECT_ROOT" --paths "$MODEL_PATHS")
if [[ "$1" == smoke ]]; then
  for model in qwen7 llama8; do
    "$PYTHON_BIN" -u -m context_attribution.frontier_probe "${args[@]}" \
      --dataset data/v2-hotpot/dev.jsonl --model-key "$model" --selection-only --limit 2 \
      --output "results/heldout-selection-smoke/$model"
  done
  exit 0
fi
model=$1
shard=$2
exec "$PYTHON_BIN" -u -m context_attribution.frontier_probe "${args[@]}" \
  --dataset data/v2-hotpot/locked_test.jsonl --model-key "$model" --heldout-shard "$shard" \
  --output "results/heldout-selection/$model/$shard"
