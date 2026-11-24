#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--project "$PROJECT_ROOT" --paths "$MODEL_PATHS" --recomp-path "$RECOMP_PATH" --model "$1")
if [[ "${2:-}" == smoke ]]; then
  args+=(--smoke --output "results/stability-smoke/$1")
else
  args+=(--dataset "$2" --shard "$3" --output "results/stability/$1/$2/$3")
fi
child=''
recover() {
  trap '' USR1 TERM
  if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
  if (( ${SLURM_RESTART_COUNT:-0} < 2 )); then
    echo "Requeueing $SLURM_JOB_ID from saved questions"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  echo 'Restart limit reached; completed questions are preserved' >&2
  exit 124
}
trap recover USR1
trap '[[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null; exit 143' TERM
for attempt in 1 2 3; do
  echo "Attempt $attempt, allocation restart ${SLURM_RESTART_COUNT:-0}"
  "$PYTHON_BIN" -u -m context_attribution.stability_study "${args[@]}" &
  child=$!
  if wait "$child"; then exit 0; else status=$?; fi
  child=''
  echo "Attempt $attempt failed with exit $status; retaining checkpoints" >&2
  sleep 5
done
exit "$status"
