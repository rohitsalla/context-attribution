#!/usr/bin/env bash
#SBATCH --job-name=context-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-11%1
#SBATCH --no-requeue
#SBATCH --output=logs/full_%A_%a.log
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ATTRIBUTION_PROGRESS=1
PYTHON="${PYTHON_BIN:-python3}"
MODEL="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
OUTPUT="${OUTPUT_DIR:-results/full-main-v1}"
# A second serial pass resumes unfinished components; completed components exit before model loading.
component=$((SLURM_ARRAY_TASK_ID % 6))
common=(--backend hf --model "$MODEL" --revision "$REVISION" --device cuda --samples 32 --resume --verify-likelihood)
case "$component" in
  0|1|2)
    caps=(128 256 512)
    cap="${caps[$component]}"
    exec "$PYTHON" -u -m context_attribution.experiment "${common[@]}" \
      --dataset data/main-v1/main.jsonl --limit 120 --caps "$cap" --output "$OUTPUT/main-$cap" ;;
  3)
    exec "$PYTHON" -u -m context_attribution.experiment "${common[@]}" \
      --dataset data/main-v1/stress.jsonl --limit 60 --caps 128 256 512 --output "$OUTPUT/stress" ;;
  4)
    exec "$PYTHON" -u -m context_attribution.experiment "${common[@]}" \
      --dataset data/main-v1/stochastic.jsonl --limit 8 --caps 128 --repeats 5 --temperature 0.7 --output "$OUTPUT/stochastic" ;;
  5)
    exec "$PYTHON" -u -m context_attribution.online --model "$MODEL" --revision "$REVISION" \
      --dataset data/main-v1/main.jsonl --output "$OUTPUT/online" ;;
esac
