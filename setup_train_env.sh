#!/bin/bash
# Builds the third environment: the GPU stack (vLLM + PyTorch) for serving models and training.
# It is kept separate from envs/legacy and envs/modern because vLLM and torch pin their own NumPy;
# sharing an env could silently change what counts as a correct migration.
#
# Run from the repo root on a GPU compute node (needs internet):
#   srun --gres=gpu:1 --time=01:00:00 --cpus-per-task=8 --pty bash
#   bash setup_train_env.sh
set -euo pipefail

module load anaconda

# 1. Python 3.12 for the GPU stack (independent of the 3.10 used by the two task runtimes)
if [ ! -x envs/py312/bin/python ]; then
  conda create -y -p envs/py312 -c conda-forge --override-channels python=3.12
fi
[ -d envs/train ] || envs/py312/bin/python -m venv envs/train
envs/train/bin/python -m pip install -q --upgrade pip

# 2. vLLM FIRST: it pins the torch version it was built against; installing torch separately
#    first usually ends in a version vLLM rejects.
envs/train/bin/python -m pip install -U vllm
envs/train/bin/python -m pip install -U transformers accelerate huggingface_hub

# 3. Record what we got
envs/train/bin/python -m pip freeze > envs/train.lock
grep -qxF "envs/py312/" .gitignore || printf "envs/py312/\nenvs/train/\n" >> .gitignore
envs/train/bin/python - <<'EOF'
import torch, vllm, transformers
print("vllm", vllm.__version__, "| torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| transformers", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT VISIBLE")
EOF
echo
echo "Driver check: the CUDA version in 'nvidia-smi' (top right) must be >= the torch CUDA version above."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
