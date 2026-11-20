#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--project "$PROJECT_ROOT" --paths "$MODEL_PATHS" --recomp-path "$RECOMP_PATH" --model "$1")
if [[ "${2:-}" == smoke ]]; then
  exec "$PYTHON_BIN" -u -m context_attribution.generalization "${args[@]}" \
    --output "results/generalization-smoke/$1"
fi
exec "$PYTHON_BIN" -u -m context_attribution.generalization "${args[@]}" \
  --dataset "$2" --shard "$3" --output "results/generalization/$1/$2/$3"
