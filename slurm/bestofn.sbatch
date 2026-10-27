#!/bin/bash
# Visible-selected best-of-k on CPU (no GPU): reruns each sample on the visible inputs.
#   RUNS="results/s1_ck00150_fb_heldout_cat results/st_s2_fb_heldout_id" sbatch slurm/bestofn.sbatch
#SBATCH --job-name=bestofn
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
: "${RUNS:?set RUNS=\"results/<run> [results/<run> ...]\"}"
echo "node: $(hostname)"
for r in $RUNS; do
  envs/legacy/bin/python -m harness.bestofn "$r" --workers 30
done
