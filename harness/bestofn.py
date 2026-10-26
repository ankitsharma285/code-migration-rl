"""Matched-compute control for the multi-turn agent: is multi-turn better than just sampling more?

Takes a SINGLE-turn run with n samples per task (e.g. the --feedback runs) and computes, for k = 1..n:
  oracle_pass@k     P(at least one of k samples passes the hidden inputs), unbiased estimator
                    (Chen et al., 2021). An upper bound: it assumes a perfect judge picks the winner.
  visible_best_of_k the fair, deployable version: draw k samples, submit the first one that passes the
                    VISIBLE examples (else the last one), score it on the hidden inputs. Same information
                    the multi-turn agent gets, without feedback going back to the model.
Compare visible_best_of_k at k = 3 against multi-turn pass at turn 3 (same number of generations; the
multi-turn agent averages fewer, since it stops early, so this comparison favours best-of-n).

  envs/legacy/bin/python -m harness.bestofn results/feedback_1.5B_train [--workers 30]
"""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

from env.feedback import feedback
from env.inputs import VISIBLE_SEEDS, make_inputs
from env.reward import compute_references, extract_code
from harness.evaluate import load_tasks
from harness.trajectory import read_jsonl


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def visible_best_of_k_exact(samples, k):
    """Exact version for small n (training-time evals): average over every ordered draw of k samples."""
    from itertools import permutations
    picks = list(permutations(range(len(samples)), k))
    tot = 0
    for p in picks:
        chosen = next((samples[i] for i in p if samples[i][0]), samples[p[-1]])
        tot += chosen[1]
    return tot / len(picks)


def visible_best_of_k(samples, k, rng, draws=2000):
    """samples: list of (visible_ok, hidden_pass). Expected hidden pass of 'first visible pass among k'."""
    tot = 0
    for _ in range(draws):
        pick = rng.sample(samples, k)
        chosen = next((s for s in pick if s[0]), pick[-1])
        tot += chosen[1]
    return tot / draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = json.load(open(run / "run.json"))["config"]
    tasks = {t["id"]: t for t in load_tasks(cfg["tasks"])}
    eps = read_jsonl(run / "trajectories.jsonl")
    if any(len(e.steps) > 1 for e in eps):
        raise SystemExit("needs a single-turn run")

    used = sorted({e.task_id for e in eps})
    vis = {i: [make_inputs(tasks[i], s) for s in VISIBLE_SEEDS] for i in used}
    with ThreadPoolExecutor(args.workers) as pool:
        vrefs = dict(zip(used, pool.map(lambda i: compute_references(tasks[i], vis[i]), used)))

        def visible_ok(e):
            code = extract_code(e.steps[-1].action)
            return code is not None and feedback(tasks[e.task_id], code, vis[e.task_id], vrefs[e.task_id])[1]
        vok = list(pool.map(visible_ok, eps))

    per_task = defaultdict(list)
    for e, v in zip(eps, vok):
        per_task[e.task_id].append((bool(v), e.reward == 1.0))
    n = min(len(s) for s in per_task.values())
    rng = random.Random(0)
    out = {"run": str(run), "tasks": len(per_task), "samples_per_task": n,
           "visible_pass_rate": round(sum(vok) / len(vok), 4),
           "hidden_pass_given_visible_pass": round(
               sum(h for s in per_task.values() for v, h in s if v) / max(sum(vok), 1), 4)}
    for k in range(1, n + 1):
        out[f"oracle_pass@{k}"] = round(sum(pass_at_k(len(s), sum(h for _, h in s), k)
                                            for s in per_task.values()) / len(per_task), 4)
        out[f"visible_best_of_{k}"] = round(sum(visible_best_of_k(s, k, rng) for s in per_task.values())
                                            / len(per_task), 4)
    (run / "bestofn.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
