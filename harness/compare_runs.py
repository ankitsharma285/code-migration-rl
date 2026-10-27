"""Paired comparison of two eval runs on the same tasks (e.g. real feedback vs placebo, trained vs base).

    python -m harness.compare_runs results/s1_mt3_heldout_id results/s1_mt3_placebo_heldout_id
    python -m harness.compare_runs results/s1_mt3_heldout_id results/st_s1_fb_heldout_id --b-best-of 3

A or B can be several runs joined by commas (e.g. two seeds): each task's score is then the mean over those runs,
so the interval pools seeds on that side.

--b-best-of K scores run B by per-task pass@K (Chen et al.) instead of mean pass: for single-turn runs this equals
visible-selected best-of-K, since a visible pass predicts a hidden pass >99% of the time (harness/bestofn.py).

Prints the final and turn-1 pass rates of each run, the difference with a 95% paired bootstrap interval
(resampling TASKS, because samples of one task are correlated), and the pass rate per migration type.
"""
import argparse
import json
import random
from math import comb
from collections import defaultdict
from pathlib import Path

from harness.evaluate import load_tasks


def pass_at_k(n, c, k):
    """Chen et al. estimator; with fewer than k samples it falls back to k = n (all samples)."""
    k = min(k, n)
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def load(run_dir, best_of=1):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "run.json"))["config"]
    per_task = defaultdict(lambda: {"final": [], "turn1": []})
    for line in open(run_dir / "trajectories.jsonl"):
        e = json.loads(line)
        if not e["steps"]:
            continue
        per_task[e["task_id"]]["final"].append(e["steps"][-1]["reason"] == "pass")
        per_task[e["task_id"]]["turn1"].append(e["steps"][0]["reason"] == "pass")
    agg = lambda v: pass_at_k(len(v), sum(v), best_of) if best_of > 1 else sum(v) / len(v)
    return cfg, {t: {k: agg(v) for k, v in d.items()} for t, d in per_task.items()}


def bootstrap(diffs, draws=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def load_many(dirs, best_of=1):
    """Several runs (comma-separated) -> per-task mean over runs, keeping only tasks present in every run."""
    loaded = [load(d, best_of) for d in dirs.split(",")]
    common = set.intersection(*(set(r) for _, r in loaded))
    out = {t: {k: sum(r[t][k] for _, r in loaded) / len(loaded) for k in ("final", "turn1")} for t in common}
    return loaded[0][0], out


def compare(a_dir, b_dir, b_best_of=1):
    cfg_a, a = load_many(a_dir)
    cfg_b, b = load_many(b_dir, b_best_of)
    common = sorted(set(a) & set(b))
    if len(common) < max(len(a), len(b)):
        print(f"warning: only {len(common)} tasks in common ({len(a)} vs {len(b)})")
    types = {t["id"]: t["types"] for t in load_tasks(cfg_a["tasks"])}
    print(f"A = {a_dir}\nB = {b_dir}" + (f"  (scored as best-of-{b_best_of})" if b_best_of > 1 else "")
          + f"\n{len(common)} tasks\n")
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--b-best-of", type=int, default=1)
    args = ap.parse_args()
    compare(args.a, args.b, args.b_best_of)
