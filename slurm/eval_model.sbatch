#!/bin/bash
# Serve a model with vLLM on one H100 and evaluate it with the harness.
#   MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct TASKS=train LIMIT=300 SAMPLES=8 OUT=results/calib_1p5b_train \
#     sbatch slurm/eval_model.sbatch
# MODEL can also be a checkpoint directory, e.g. results/grpo_s1/ckpt-00300
# FEWSHOT=2 adds two hand-written demos to every prompt (few-shot baseline)
# TURNS=3 runs multi-turn episodes with automatic execution feedback (v2); MAXLEN sets the context (default 8192)
# FEEDBACK=1 adds the legacy code's errors/warnings on the visible examples (execution-feedback test)
#SBATCH --job-name=eval
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
# --generation-config vllm: ignore the model's generation_config.json (top_k 20, top_p 0.8, repetition_penalty 1.1),
# so evaluation samples exactly like training (plain temperature sampling, train/generation.py).
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
if [ -n "${CUDA_MODULE:-}" ]; then module load "$CUDA_MODULE"; fi
export PATH="$PWD/envs/train/bin:$PATH"
if command -v nvcc > /dev/null; then export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
else export VLLM_USE_FLASHINFER_SAMPLER=0; fi
: "${MODEL:?}" "${OUT:?}"
TASKS="${TASKS:-heldout_id}"; SAMPLES="${SAMPLES:-8}"; TEMP="${TEMP:-1.0}"; SEEDS="${SEEDS:-train}"
PORT=$((8000 + SLURM_JOB_ID % 1000))

envs/train/bin/vllm serve "$MODEL" --served-model-name policy --dtype bfloat16 --max-model-len "${MAXLEN:-8192}" \
  --gpu-memory-utilization 0.85 --port "$PORT" \
  --generation-config vllm > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null; then echo "vLLM ready after ~$((i * 10))s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; fi
  sleep 10
done

envs/legacy/bin/python -m harness.evaluate --policy openai --model policy --base-url "http://localhost:$PORT/v1" \
  --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"} --samples "$SAMPLES" --temperature "$TEMP" --seeds "$SEEDS" \
  ${FEWSHOT:+--fewshot "$FEWSHOT"} ${FEEDBACK:+--feedback} ${TURNS:+--turns "$TURNS"} \
  --gen-workers 64 --workers 30 --out "$OUT"
