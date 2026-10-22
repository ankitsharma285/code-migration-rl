#!/bin/bash
# Builds the two task runtimes for the code-migration RL project.
#
# Run from the project folder on a compute node (no GPU needed):
#   srun --time=01:00:00 --pty bash
#   cd /path/to/code-migration-rl
#   bash setup_envs.sh
#
# Result:
#   envs/py310/   bare Python 3.10 from conda (the venvs below depend on it; don't delete)
#   envs/legacy/  pandas 1.3.5 + NumPy 1.21.6
#   envs/modern/  pandas 2.2.3 + NumPy 2.2.6
#   envs/*.lock   pinned package lists to commit to git
set -euo pipefail

module load anaconda

# 1. A bare Python 3.10 interpreter.
#    conda-forge with --override-channels avoids the Anaconda "defaults" channel
#    terms-of-service prompt that newer conda versions raise in scripts.
if [ ! -x envs/py310/bin/python ]; then
  conda create -y -p envs/py310 -c conda-forge --override-channels python=3.10
fi
PY310="$PWD/envs/py310/bin/python"
"$PY310" --version

# 2. Two plain venvs built from that interpreter.
#    --only-binary makes pip fail fast instead of compiling pandas from source.
for name in legacy modern; do
  [ -d "envs/$name" ] || "$PY310" -m venv "envs/$name"
  "envs/$name/bin/python" -m pip install -q --upgrade pip
done
envs/legacy/bin/python -m pip install --only-binary=:all: pandas==1.3.5 numpy==1.21.6 "python-dateutil==2.8.2"
envs/modern/bin/python -m pip install --only-binary=:all: pandas==2.2.3 numpy==2.2.6

# 3. Lockfiles (commit these; the env folders stay out of git).
envs/legacy/bin/python -m pip freeze > envs/legacy.lock
envs/modern/bin/python -m pip freeze > envs/modern.lock
grep -qxF "envs/py310/" .gitignore 2>/dev/null || printf "envs/py310/\nenvs/legacy/\nenvs/modern/\n" >> .gitignore

# 4. Smoke test. Expected:
#    legacy: append OK, np.float WARNS (deprecated since NumPy 1.20), applymap OK
#    modern: append FAILS, np.float FAILS, applymap WARNS (FutureWarning)
cat > envs/smoke.py <<'EOF'
import warnings, pandas as pd, numpy as np
print(f"pandas {pd.__version__}, numpy {np.__version__}")
df = pd.DataFrame({"a": [1]})
checks = [("append", lambda: df.append(df)),
          ("np.float", lambda: np.float(1)),
          ("applymap", lambda: df.applymap(str))]
for name, fn in checks:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            print(f"  {name:9s} {'WARNS: ' + w[0].category.__name__ if w else 'OK'}")
        except Exception as e:
            print(f"  {name:9s} FAILS: {type(e).__name__}")
EOF
for name in legacy modern; do
  echo "== $name"
  "envs/$name/bin/python" envs/smoke.py
done
