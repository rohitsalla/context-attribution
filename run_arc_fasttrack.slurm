#!/usr/bin/env bash
#SBATCH --job-name=context-attribution-cont
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/pilot_continue_%A_%a.log
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ATTRIBUTION_PROGRESS=1
PYTHON="${PYTHON_BIN:-python3}"
MODEL="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
OUTPUT="${OUTPUT_DIR:-results/qwen25-7b-arc}"
exec "$PYTHON" -u -m context_attribution.experiment --backend hf --model "$MODEL" \
  --revision "$REVISION" --device cuda \
  --limit 20 --caps 128 256 --samples 32 --resume --output "$OUTPUT"
