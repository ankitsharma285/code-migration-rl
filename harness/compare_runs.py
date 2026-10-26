"""Paired comparison of two eval runs on the same tasks (e.g. real feedback vs placebo, trained vs base).

    python -m harness.compare_runs results/s1_mt3_heldout_id results/s1_mt3_placebo_heldout_id

Prints the final and turn-1 pass rates of each run, the difference with a 95% paired bootstrap interval
(resampling TASKS, because samples of one task are correlated), and the pass rate per migration type.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from harness.evaluate import load_tasks


def load(run_dir):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "run.json"))["config"]
    per_task = defaultdict(lambda: {"final": [], "turn1": []})
    for line in open(run_dir / "trajectories.jsonl"):
        e = json.loads(line)
        if not e["steps"]:
            continue
        per_task[e["task_id"]]["final"].append(e["steps"][-1]["reason"] == "pass")
        per_task[e["task_id"]]["turn1"].append(e["steps"][0]["reason"] == "pass")
    return cfg, {t: {k: sum(v) / len(v) for k, v in d.items()} for t, d in per_task.items()}


def bootstrap(diffs, draws=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def compare(a_dir, b_dir):
    cfg_a, a = load(a_dir)
    cfg_b, b = load(b_dir)
    common = sorted(set(a) & set(b))
    if len(common) < max(len(a), len(b)):
        print(f"warning: only {len(common)} tasks in common ({len(a)} vs {len(b)})")
    types = {t["id"]: t["types"] for t in load_tasks(cfg_a["tasks"])}
    print(f"A = {a_dir}\nB = {b_dir}\n{len(common)} tasks\n")
    for metric in ("turn1", "final"):
        ma = sum(a[t][metric] for t in common) / len(common)
        mb = sum(b[t][metric] for t in common) / len(common)
        diffs = [a[t][metric] - b[t][metric] for t in common]
        lo, hi = bootstrap(diffs)
        print(f"{metric:6s}  A {ma:6.1%}   B {mb:6.1%}   A-B {ma - mb:+6.1%}   95% CI [{lo:+.1%}, {hi:+.1%}]")
    by_type = defaultdict(list)
    for t in common:
        for ty in types.get(t, ["?"]):
            by_type[ty].append(t)
    print("\nfinal pass by type:            n      A      B    A-B")
    for ty, ts in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        ma = sum(a[t]["final"] for t in ts) / len(ts)
        mb = sum(b[t]["final"] for t in ts) / len(ts)
        print(f"  {ty:28s} {len(ts):4d}  {ma:5.0%}  {mb:5.0%}  {ma - mb:+5.0%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    compare(sys.argv[1], sys.argv[2])
