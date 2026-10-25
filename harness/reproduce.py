"""Reproduce a result from its record: check out the recorded commit, rerun, compare.

Usage (from the repo root):  envs/legacy/bin/python -m harness.reproduce results/<run>

Steps:
  1. Refuse records made from a dirty tree (their code is not in any commit).
  2. Check out the recorded commit into a temporary git worktree (your working copy is untouched).
  3. Rerun there with the recorded config, using this repo's pinned envs (envs/ is not in git; the
     package freezes are compared instead, so a changed env is reported, not silently accepted).
     Deterministic policies are re-sampled. Model policies are RE-SCORED from the stored
     completions, which reproduces the grading exactly without calling the model again.
  4. Compare metrics.json (the deterministic summary) and the task-set hash.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from env.runner import REPO, envs_dir
from harness.manifest import env_freeze

CONFIG_FLAGS = ["policy", "tasks", "limit", "seeds", "samples", "model", "base_url", "api_key_env",
                "temperature", "max_tokens", "workers", "gen_workers", "fewshot", "feedback"]


def to_argv(config, rescore_path=None):
    argv = []
    for k in CONFIG_FLAGS:
        v = config.get(k)
        if v is None or v is False:
            continue
        flag = f"--{k.replace('_', '-')}"
        argv += [flag] if v is True else [flag, str(v)]
    if rescore_path:
        argv += ["--rescore", str(rescore_path)]
    return argv


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    record = json.loads((run_dir / "run.json").read_text())
    if record["git"]["dirty"]:
        sys.exit("Not reproducible: this result was produced from uncommitted code.")
    commit = record["git"]["commit"]
    scripted = record["policy"] in ("oracle", "legacy") or record["policy"].startswith("cheater:")
    print(f"reproducing {run_dir.name}: policy {record['policy']}, commit {commit[:8]}, "
          f"{'re-sampling' if scripted else 're-scoring stored completions'}")

    for rt in ("legacy", "modern"):
        now = env_freeze(rt)["sha256"]
        if now != record["envs"][rt]["sha256"]:
            print(f"WARNING: the {rt} env's packages differ from the record (results may differ)")

    work = Path(tempfile.mkdtemp(prefix="repro_"))
    tree = work / "tree"
    subprocess.run(["git", "worktree", "add", "--detach", str(tree), commit], cwd=REPO, check=True,
                   capture_output=True)
    try:
        out = work / "out"
        rescore = None if scripted else run_dir / "trajectories.jsonl"
        cmd = [sys.executable, "-m", "harness.evaluate", *to_argv(record["config"], rescore), "--out", str(out)]
        env = {**os.environ, "CMRL_ENVS": str(envs_dir().resolve())}
        proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"rerun failed:\n{proc.stderr[-2000:]}")
        new_record = json.loads((out / "run.json").read_text())
        old_m = json.loads((run_dir / "metrics.json").read_text())
        new_m = json.loads((out / "metrics.json").read_text())
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=REPO, capture_output=True)

    same_tasks = new_record["task_set_sha256"] == record["task_set_sha256"]
    diffs = [k for k in sorted(set(old_m) | set(new_m)) if old_m.get(k) != new_m.get(k)]
    print(f"task set hash: {'match' if same_tasks else 'DIFFERENT'}")
    print(f"metrics: {'identical' if not diffs else 'DIFFER in ' + ', '.join(diffs)}")
    print(f"pass@1 recorded {old_m['pass_at_1']:.4f}, reproduced {new_m['pass_at_1']:.4f}")
    ok = same_tasks and not diffs
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
