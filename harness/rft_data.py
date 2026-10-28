"""Stage 8 data for the RFT baseline (rejection-sampling fine-tuning): sample the BASE model on the training
tasks, keep the completions that pass the hidden inputs, fine-tune on them (train/train_sft.py).

Compute matching. The single-turn GRPO control generated prompts_per_step x group_size x max_steps =
64 x 8 x 150 = 76,800 completions. Sampling the base model 44 times on each of the 1,762 training tasks
gives 77,528, the same budget, with the same prompt (turn 1 with feedback on the legacy code), the same
temperature (1.0), the same hidden inputs (TRAIN_HIDDEN_SEEDS) and the same reward. Only the learning
rule differs: GRPO re-samples from the current policy and pushes failures down; RFT samples once from the
base model and imitates the successes.

  # 1. split the training tasks into shards, one GPU job each (sampling + scoring 77k completions is long)
  envs/legacy/bin/python -m harness.rft_data shard --split train --n 8 --out results/rft/shards
  # 2. per shard: MODEL=<base> TASKS=results/rft/shards/train_<k>.jsonl SAMPLES=44 FEEDBACK=1 \
  #      OUT=results/rft/sample_<k> sbatch slurm/eval_model.sbatch
  # 3. keep the passing completions
  envs/legacy/bin/python -m harness.rft_data build --runs 'results/rft/sample_*' --out results/rft/sft_data.jsonl

build keeps reward == 1 completions, drops exact duplicates within a task (after stripping whitespace),
and keeps at most --cap per task (default 8, the GRPO group size), chosen at random with a fixed seed, so
tasks the base model already solves every time do not dominate the data.
"""
import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from harness.evaluate import load_tasks


def shard(args):
    tasks = load_tasks(args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.split).stem if args.split.endswith(".jsonl") else args.split
    for k in range(args.n):
        part = tasks[k::args.n]
        path = out / f"{name}_{k}.jsonl"
        with open(path, "w") as f:
            for t in part:
                f.write(json.dumps(t) + "\n")
        print(f"{path}: {len(part)} tasks")


def build(args):
    runs = sorted(d for pat in args.runs.split(",") for d in glob.glob(pat))
    if not runs:
        raise SystemExit(f"no run directories match {args.runs!r}")
    passing = defaultdict(list)
    n_eps, tasks_seen, commits = 0, set(), set()
    for d in runs:
        commits.add(json.load(open(Path(d, "run.json")))["git"]["commit"])
        for line in open(Path(d, "trajectories.jsonl")):
            e = json.loads(line)
            n_eps += 1
            tasks_seen.add(e["task_id"])
            if len(e["steps"]) != 1:
                raise SystemExit(f"{d}: multi-turn episode; RFT samples single-turn (do not set TURNS)")
            st = e["steps"][0]
            if st["reason"] == "pass":
                passing[e["task_id"]].append((e["sample_idx"], st["prompt"], st["action"]))
    rng = random.Random(args.seed)
    rows, dupes = [], 0
    per_task = Counter()
    for tid in sorted(passing):
        seen, uniq = set(), []
        for idx, msgs, comp in sorted(passing[tid], key=lambda x: x[0]):
            key = "\n".join(l.rstrip() for l in comp.strip().splitlines())
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            uniq.append((msgs, comp))
        rng.shuffle(uniq)
        for msgs, comp in uniq[: args.cap]:
            rows.append({"task_id": tid, "messages": msgs, "completion": comp})
            per_task[tid] += 1
    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_pass = sum(len(v) for v in passing.values())
    stats = {"runs": runs, "commits": sorted(c or "none" for c in commits), "episodes": n_eps,
             "tasks_sampled": len(tasks_seen), "passing_episodes": n_pass,
             "pass_rate": round(n_pass / max(n_eps, 1), 4), "tasks_with_a_pass": len(passing),
             "exact_duplicates_dropped": dupes, "cap_per_task": args.cap, "examples": len(rows),
             "examples_per_task_hist": dict(sorted(Counter(per_task.values()).items()))}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "runs"}, indent=1))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--split", default="train")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--out", default="results/rft/shards")
    b = sub.add_parser("build")
    b.add_argument("--runs", required=True, help="glob(s) of sampling run dirs, comma-separated")
    b.add_argument("--out", default="results/rft/sft_data.jsonl")
    b.add_argument("--cap", type=int, default=8)
    b.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    shard(args) if args.cmd == "shard" else build(args)


if __name__ == "__main__":
    main()
