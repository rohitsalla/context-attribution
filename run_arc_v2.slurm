#!/usr/bin/env bash
#SBATCH --job-name=context-v2-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=00:55:00
#SBATCH --output=logs/v2_%A_%a.log
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
extra=()
if [[ "${PREFLIGHT:-0}" == 1 ]]; then extra+=(--preflight); fi
exec "${PYTHON_BIN:-python3}" -u -m context_attribution.v2.launch \
  --paths "${MODEL_PATHS:?Set private model-path registry}" \
  --task "${SLURM_ARRAY_TASK_ID:-0}" "${extra[@]}"
