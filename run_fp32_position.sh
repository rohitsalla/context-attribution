#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
if ! "${PYTHON_BIN:?}" - <<'PY'
import json
from pathlib import Path
path=Path('results/precision-probe/completed.json')
try:
    result=json.loads(path.read_text())
    passed=result['complete'] and result['by_precision']['float32']['all_4d_checks_within_1e_minus4']
except (OSError,KeyError,ValueError):passed=False
if not passed:
    output=Path('results/position-probe-fp32');output.mkdir(parents=True,exist_ok=True)
    (output/'skipped.json').write_text(json.dumps({'skipped':True,'reason':'CUDA FP32 equivalence gate has not passed'}))
    raise SystemExit(1)
PY
then
  exit 0
fi
exec "$PYTHON_BIN" -u -m "${FP32_POSITION_MODULE:-context_attribution.position_probe}" \
  --root "$PROJECT_ROOT" --paths "${MODEL_PATHS:?}" \
  --reference results/v4/evidence_hotpot_qwen7/traces.jsonl \
  --dtype float32 --force-math --output results/position-probe-fp32
