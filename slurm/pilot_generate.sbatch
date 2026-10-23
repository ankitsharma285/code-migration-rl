#!/bin/bash
# Pilot task generation: serve Gemma 4 31B on one GPU, generate N tasks, filter them.
# Submit from the repo root:   sbatch slurm/pilot_generate.sbatch
# Larger runs:                 N=2000 SEED=1 sbatch --time=08:00:00 slurm/pilot_generate.sbatch
#
# GPU: requests an H100 by name (a bare gpu:1 can land on a 16 GB V100). Uses the default partition.
#SBATCH --job-name=gemma-gen
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=16              # the filter's sandboxes run on these cores
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs tasks/generated

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"     # model weights (~62 GB, few files) live outside the repo

# FlashInfer JIT-compiles its sampling kernel on first use and needs nvcc (the CUDA toolkit).
# Use the cluster's CUDA module if one is set (CUDA_MODULE=cuda/13.0 sbatch ...); otherwise fall back
# to vLLM's PyTorch sampler, which needs no compiler (slightly slower sampling, same outputs).
if [ -n "${CUDA_MODULE:-}" ]; then module load "$CUDA_MODULE"; fi
if command -v nvcc > /dev/null; then
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
  echo "nvcc found: $(nvcc --version | tail -1)"
else
  export VLLM_USE_FLASHINFER_SAMPLER=0
  echo "no nvcc: using vLLM's PyTorch sampler (VLLM_USE_FLASHINFER_SAMPLER=0)"
fi
N="${N:-50}"
SEED="${SEED:-0}"
SPLIT="${SPLIT:-train}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
RUN="${SPLIT}_s${SEED}_n${N}_${SLURM_JOB_ID}"

# 1. Serve the generator. FP8 weights (~31 GB) leave room for the KV cache on one 80 GB H100;
#    bf16 weights (~62 GB) would leave almost none. Text only: skip multimodal profiling.
envs/train/bin/vllm serve google/gemma-4-31B-it \
  --served-model-name gemma4 \
  --quantization fp8_per_tensor \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --port "$PORT" > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "waiting for vLLM on port $PORT (first run also downloads the weights)..."
for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null; then echo "vLLM ready after ~$((i * 10))s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; fi
  sleep 10
done

# 2. Generate, then 3. filter (the sandbox runs on the CPU cores; the GPU is idle during filtering)
envs/legacy/bin/python -m tasks.generate --base-url "http://localhost:$PORT/v1" --model gemma4 \
  --n "$N" --seed "$SEED" --split "$SPLIT" --out "tasks/generated/raw_$RUN.jsonl"
kill $VLLM_PID 2>/dev/null || true
envs/legacy/bin/python -m tasks.filter_tasks "tasks/generated/raw_$RUN.jsonl" --out "tasks/generated/$RUN"
echo "done: tasks/generated/$RUN/{accepted.jsonl,rejected.jsonl,stats.json}"
