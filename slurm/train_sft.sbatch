#!/bin/bash
# SFT on filtered successes (Stage 8 RFT baseline; data from harness/rft_data.py build).
#   CONFIG=configs/rft_1p5b.json OUT=results/rft_s1 sbatch slurm/train_sft.sbatch
#   SMOKE=1 CONFIG=configs/rft_1p5b.json OUT=results/rft_smoke sbatch slurm/train_sft.sbatch   # 64 examples, 1 epoch
#SBATCH --job-name=sft
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  job $SLURM_JOB_ID"
nvidia-smi -L || { echo "nvidia-smi failed on $(hostname): resubmit with --exclude=$(hostname)"; exit 1; }
envs/train/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "torch cannot use the GPU on $(hostname): resubmit with --exclude=$(hostname)"; exit 1; }
: "${CONFIG:?}" "${OUT:?}"
envs/train/bin/python -m train.train_sft --config "$CONFIG" --out "$OUT" ${SMOKE:+--set limit=64 epochs=1}
