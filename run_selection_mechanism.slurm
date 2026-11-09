#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${1:-}" == smoke ]]; then
  for study in hotpot_qwen7 musique_qwen7; do
    "$PYTHON_BIN" -u -m context_attribution.selection_mechanism --root "$PROJECT_ROOT" --paths "$MODEL_PATHS" \
      --study "$study" --limit 2 --output "results/selection-mechanism-smoke/$study"
  done
  exit 0
fi
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
for study in ['hotpot_qwen7','musique_qwen7']:
    assert json.loads((Path('results/selection-mechanism-smoke')/study/'completed.json').read_text())['n']==2
PY
studies=(hotpot_qwen7 hotpot_llama8 musique_qwen7 musique_llama8)
study=${studies[${SLURM_ARRAY_TASK_ID:?}]}
exec "$PYTHON_BIN" -u -m context_attribution.selection_mechanism --root "$PROJECT_ROOT" --paths "$MODEL_PATHS" \
  --study "$study" --output "results/selection-mechanism/$study"
