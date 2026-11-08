#!/usr/bin/env bash
#SBATCH --job-name=context-v4-controlled
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=00:25:00
#SBATCH --output=logs/v4_%A_%a.log
set -euo pipefail
cd "${PROJECT_ROOT:?Set new frozen workspace}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
task="${SLURM_ARRAY_TASK_ID:-0}"
if (( task <= 4 )); then
  architecture=anchored; seed=1729; extra=()
  if (( task == 0 )); then extra=(--smoke --train-limit 2 --dev-limit 2 --warmup-steps 2 --steps 2); fi
  if (( task == 1 )); then architecture=pure; fi
  if (( task >= 2 )); then seed=$((1729+task-2)); fi
  exec "${PYTHON_BIN:?}" -u -m context_attribution.v4.branches --root "$PWD" \
    --paths "${MODEL_PATHS:?}" --architecture "$architecture" --seed "$seed" \
    --output "results/v4/task${task}_${architecture}_${seed}" "${extra[@]}"
fi
models=(qwen7 llama8 mistral7)
if (( task <= 10 )); then
  index=$((task-5)); key="${models[$((index%3))]}"; dataset=data/v2-hotpot/dev.jsonl; name=hotpot
  if (( index >= 3 )); then dataset=data/v4-musique/dev.jsonl; name=musique; fi
  exec "$PYTHON_BIN" -u -m context_attribution.v4.evidence --root "$PWD" --paths "$MODEL_PATHS" \
    --dataset "$dataset" --model-key "$key" --output "results/v4/evidence_${name}_${key}"
fi
if (( task <= 13 )); then
  key="${models[$((task-11))]}"
  exec "$PYTHON_BIN" -u -m context_attribution.v4.generative --root "$PWD" --paths "$MODEL_PATHS" \
    --model-key "$key" --output "results/v4/strong_global_${key}"
fi
echo 'Invalid task' >&2;exit 2
