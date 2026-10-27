#!/bin/bash
# Re-score stored completions on the evaluation hidden seeds (CPU only).
#   RUNS="results/a results/b" sbatch slurm/rescore.sbatch
#SBATCH --job-name=rescore
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
: "${RUNS:?set RUNS=\"results/<run> ...\"}"
echo "node: $(hostname)"
envs/legacy/bin/python -m harness.rescore_runs $RUNS --seeds "${SEEDS:-eval:final}" --out-dir "${OUT_DIR:-results/final}" --workers 30
