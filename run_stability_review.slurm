#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" PYTHONHASHSEED=0
exec "$PYTHON_BIN" -u review_stability.py --project "$PROJECT_ROOT"
