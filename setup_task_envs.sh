#!/bin/bash
# Builds the two task runtimes from the committed lock files:
#   envs/legacy  pandas 1.3 / NumPy 1.21  -> reference outputs (the behavior to preserve)
#   envs/modern  pandas 2.2 / NumPy 2.x   -> where candidate migrations run
# Both use Python 3.10 (the newest Python pandas 1.3.5 supports). Run from the repo root; needs internet.
set -euo pipefail

if [ ! -x envs/py310/bin/python ]; then
  if command -v conda > /dev/null; then
    conda create -y -p envs/py310 -c conda-forge --override-channels python=3.10
  else
    echo "Need a Python 3.10 at envs/py310/bin/python (conda not found): install one there and rerun"; exit 1
  fi
fi
for rt in legacy modern; do
  [ -d "envs/$rt" ] || envs/py310/bin/python -m venv "envs/$rt"
  "envs/$rt/bin/python" -m pip install -q --upgrade pip
  "envs/$rt/bin/python" -m pip install -q -r "envs/$rt.lock"
  "envs/$rt/bin/python" -c "import pandas, numpy; print('$rt: pandas', pandas.__version__, '| numpy', numpy.__version__)"
done
