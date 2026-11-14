#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--root "$PROJECT_ROOT" --paths "$MODEL_PATHS" --dataset data/v2-hotpot/dev.jsonl
      --selection-only --recomp-path "${RECOMP_PATH:?}")
if [[ "$1" == smoke ]]; then
  for model in qwen7 llama8; do
    "$PYTHON_BIN" -u -m context_attribution.frontier_probe "${args[@]}" \
      --model-key "$model" --limit 2 --output "results/recomp-smoke/$model"
  done
  exit 0
fi
exec "$PYTHON_BIN" -u -m context_attribution.frontier_probe "${args[@]}" \
  --model-key "$1" --output "results/recomp-selection/hotpot_$1"
