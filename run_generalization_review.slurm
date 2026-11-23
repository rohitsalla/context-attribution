#!/usr/bin/env bash
#SBATCH --account=tealab
#SBATCH --partition=normal_q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ctx-generalization-review
#SBATCH --output=logs/review_%j.log
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONHASHSEED=0
exec "${PYTHON_BIN:-python3}" -u review_generalization.py \
  --project "$PWD" --results "$PWD/results/generalization"
