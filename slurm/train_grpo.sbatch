#!/bin/bash
# GRPO training on ONE H100 (vLLM rollouts and the trainer share the GPU).
#   CONFIG=configs/grpo_1p5b.json OUT=results/grpo_s1 sbatch slurm/train_grpo.sbatch
#   RESUME=1 OUT=results/grpo_s1 sbatch slurm/train_grpo.sbatch        # continue after the time limit
#SBATCH --job-name=grpo
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=32               # the sandbox scores completions on these cores
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0  # vLLM in this process, so the trainer can sync weights into it
if [ -n "${CUDA_MODULE:-}" ]; then module load "$CUDA_MODULE"; fi
command -v nvcc > /dev/null || export VLLM_USE_FLASHINFER_SAMPLER=0
# fail fast, with the node named, if this node's GPU is unusable (a bad node otherwise dies inside vLLM)
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  job $SLURM_JOB_ID"
nvidia-smi -L || { echo "nvidia-smi failed on $(hostname): resubmit with --exclude=$(hostname)"; exit 1; }
envs/train/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "torch cannot use the GPU on $(hostname): resubmit with --exclude=$(hostname)"; exit 1; }
: "${OUT:?set OUT=results/<run name>}"
if [ -n "${RESUME:-}" ]; then
  envs/train/bin/python -m train.train_grpo --out "$OUT" --resume
else
  : "${CONFIG:?set CONFIG=configs/<file>.json}"
  envs/train/bin/python -m train.train_grpo --config "$CONFIG" --out "$OUT"
fi
