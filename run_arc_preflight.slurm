#!/usr/bin/env bash
#SBATCH --job-name=context-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/full_preflight_%j.log
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ATTRIBUTION_PROGRESS=1
exec "${PYTHON_BIN:-python3}" -u -m context_attribution.experiment --backend hf \
  --model "${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}" \
  --revision "${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}" --device cuda \
  --dataset data/main-v1/main.jsonl --limit 1 --caps 128 --verify-likelihood \
  --output "${OUTPUT_DIR:-results/full-main-v1}/preflight"
