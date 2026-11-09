#!/usr/bin/env bash
# Resource limits and dependency are supplied explicitly at submission.
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export PYTHONPATH="$PROJECT_ROOT/src"
if [[ "${1:-}" == smoke ]]; then
  for benchmark in hotpot musique; do
    dataset=data/v2-hotpot/dev.jsonl
    if [[ "$benchmark" == musique ]]; then dataset=data/v4-musique/dev.jsonl; fi
    "$PYTHON_BIN" -u -m context_attribution.frontier_probe --root "$PROJECT_ROOT" --paths "$MODEL_PATHS" \
      --dataset "$dataset" --model-key qwen7 --limit 2 --output "results/frontier-smoke/$benchmark"
  done
  exit 0
fi
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
for benchmark in ['hotpot','musique']:
    root=Path('results/frontier-smoke')/benchmark
    assert json.loads((root/'completed.json').read_text())['n']==2
    rows=[json.loads(s) for s in (root/'traces.jsonl').read_text().splitlines()]
    assert len(rows)==12 and all(r['same_shape_repeat_error']<=1e-7 for r in rows)
    assert all(r['raw_identity_endpoint_error']<=1e-7 for r in rows if r['method']=='raw')
PY
task=${SLURM_ARRAY_TASK_ID:?}
models=(qwen7 llama8)
key=${models[$((task%2))]}
benchmark=hotpot;dataset=data/v2-hotpot/dev.jsonl
if (( task >= 2 )); then benchmark=musique;dataset=data/v4-musique/dev.jsonl; fi
exec "$PYTHON_BIN" -u -m context_attribution.frontier_probe --root "$PROJECT_ROOT" --paths "$MODEL_PATHS" \
  --dataset "$dataset" --model-key "$key" --output "results/frontier/${benchmark}_${key}"
