#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--project "$PROJECT_ROOT" --paths "$MODEL_PATHS" --archive "$ARCHIVE_RESULTS" --recomp-path "$RECOMP_PATH")
if [[ "$1" == smoke ]]; then
  for model in qwen7 llama8; do
    "$PYTHON_BIN" -u -m context_attribution.heldout_followup "${args[@]}" \
      --model "$model" --output "results/heldout-followup-smoke/$model"
  done
  exit 0
fi
exec "$PYTHON_BIN" -u -m context_attribution.heldout_followup "${args[@]}" \
  --model "$1" --shard "$2" --output "results/heldout-followup/$1/$2"
