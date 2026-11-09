#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export PYTHONPATH="$PROJECT_ROOT/src"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
for name in ['hotpot','musique']:
    assert json.loads((Path('results/frontier-smoke')/name/'completed.json').read_text())['n']==2
PY
models=(qwen7 llama8)
key=${models[${SLURM_ARRAY_TASK_ID:?}]}
exec "$PYTHON_BIN" -u -m context_attribution.frontier_probe --root "$PROJECT_ROOT" --paths "$MODEL_PATHS" \
  --dataset data/v4-musique/frontier32.jsonl --model-key "$key" --limit 32 --output "results/frontier/musique_${key}"
