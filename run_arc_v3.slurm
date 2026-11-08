#!/usr/bin/env bash
#SBATCH --job-name=context-v3-followup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=00:25:00
#SBATCH --output=logs/v3_%A_%a.log
set -euo pipefail
cd "${PROJECT_ROOT:?Set frozen follow-up root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
task="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$task" == 13 ]]; then
  exec "${PYTHON_BIN:?}" -u -m context_attribution.v3.soft_study --root "$PWD" \
    --paths "${MODEL_PATHS:?}" --mode unpaired --slots 32 --train-limit 2 --dev-limit 2 \
    --warmup-steps 4 --steps 4 --output results/v3/training_smoke
fi
if [[ "$task" == 0 ]]; then
  exec "${PYTHON_BIN:?}" -u -m context_attribution.v3.soft_study --root "$PWD" \
    --paths "${MODEL_PATHS:?}" --mode canaries --dev-limit 32 --output results/v3/canary_eval
fi
if (( task >= 1 && task <= 9 )); then
  modes=(factual unpaired paired)
  mode="${modes[$(((task-1)%3))]}"
  seed=$((1729+(task-1)/3))
  exec "$PYTHON_BIN" -u -m context_attribution.v3.soft_study --root "$PWD" \
    --paths "$MODEL_PATHS" --mode "$mode" --seed "$seed" --slots 32 \
    --train-limit 128 --warmup-steps 256 --steps 512 --dev-limit 32 \
    --output "results/v3/${mode}_${seed}"
fi
if (( task >= 10 && task <= 12 )); then
  exec "$PYTHON_BIN" -u -m context_attribution.v3.expand_cheap \
    --root "$PWD" --paths "$MODEL_PATHS" --profile "$((task-10))"
fi
echo "Unknown task $task" >&2; exit 2
