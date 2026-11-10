#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${1:-}" == smoke ]]; then
  for study in hotpot_qwen7 musique_llama8; do
    "$PYTHON_BIN" -u -m context_attribution.replacement_layout --root "$PROJECT_ROOT" --paths "$MODEL_PATHS" \
      --study "$study" --smoke --output "results/replacement-layout-smoke/$study"
  done
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from context_attribution.replacement_layout import SHAPE_TOLERANCE, read_rows

for study in ("hotpot_qwen7", "musique_llama8"):
    folder = Path("results/replacement-layout-smoke") / study
    assert json.loads((folder / "completed.json").read_text())["n"] == 2
    rows = read_rows(folder / "traces.jsonl")
    assert any(row["mask_error"] is not None for row in rows)
    assert max(branch["shape_error"] for row in rows for branch in row["branches"].values()) <= SHAPE_TOLERANCE
PY
  exit 0
fi
studies=(hotpot_qwen7 hotpot_llama8 musique_qwen7 musique_llama8)
study=${studies[${SLURM_ARRAY_TASK_ID:?}]}
exec "$PYTHON_BIN" -u -m context_attribution.replacement_layout --root "$PROJECT_ROOT" --paths "$MODEL_PATHS" \
  --study "$study" --output "results/replacement-layout/$study"
