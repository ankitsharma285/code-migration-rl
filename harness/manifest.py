"""Reproducibility record stored with every result (run.json).

A result is reproducible from its record when it names: the git commit (and that the tree was
clean), the exact packages in both runtimes, a hash of the task set, the seeds, and the full
config. harness/reproduce.py reruns from exactly these.
"""
import hashlib
import json
import platform
import socket
import subprocess
import time

from env.runner import REPO, python_for


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    except OSError:
        return ""


def git_state():
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit or None, "dirty": bool(status) or not commit,
            "dirty_files": status.splitlines()[:20]}


def env_freeze(runtime):
    out = subprocess.run([python_for(runtime), "-m", "pip", "freeze"], capture_output=True, text=True,
                         timeout=120).stdout
    return {"packages": out.splitlines(), "sha256": hashlib.sha256(out.encode()).hexdigest()}


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build(config, tasks):
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(),
        "envs": {rt: env_freeze(rt) for rt in ("legacy", "modern")},
        "task_set_sha256": sha256_json(tasks),
        "n_tasks": len(tasks),
        "config": config,
        "host": {"hostname": socket.gethostname(), "python": platform.python_version()},
    }
