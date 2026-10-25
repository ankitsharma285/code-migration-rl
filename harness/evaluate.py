"""Evaluate any policy on a task set; write trajectories, metrics, and a reproducibility record.

Examples (from the repo root):
  envs/legacy/bin/python -m harness.evaluate --policy oracle --out results/oracle_seed
  envs/legacy/bin/python -m harness.evaluate --policy cheater:private_append --out results/cheat
  envs/legacy/bin/python -m harness.evaluate --policy openai --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --base-url http://localhost:8000/v1 --samples 8 --out results/base_1p5b
  # re-score completions stored in an earlier run (reproduces a result without calling the model):
  envs/legacy/bin/python -m harness.evaluate --rescore results/base_1p5b/trajectories.jsonl --out results/rescore

Outputs in --out:
  run.json           reproducibility record (commit, dirty flag, package freezes, task hash, config)
  trajectories.jsonl one Episode per (task, sample)
  metrics.json       deterministic summary (no timings) -- what reproduce.py compares
  timing.json        wall-clock breakdown (generation vs. reference vs. scoring), for profiling
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from env import prompt as prompt_mod
from env.inputs import TRAIN_HIDDEN_SEEDS, VISIBLE_SEEDS, eval_seeds, make_inputs
from env.reward import compute_references, score
from env.runner import Limits
from harness import manifest
from harness.cheaters import CHEATERS
from harness.policies import LEGACY, ORACLE, OpenAIPolicy, ScriptedPolicy
from harness.trajectory import Episode, Step, read_jsonl, write_jsonl


def load_tasks(name):
    """'seed' (hand-written), a frozen split name (train | heldout_id | heldout_cat), or a .jsonl path."""
    if name == "seed":
        from tasks.seed_tasks import SEED_TASKS
        return SEED_TASKS
    path = Path(name) if name.endswith(".jsonl") else Path(__file__).resolve().parent.parent / "tasks" / "splits" / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"unknown task set {name!r}: no {path}")
    return [json.loads(line) for line in open(path)]


def make_policy(args):
    if args.policy == "oracle":
        return ORACLE
    if args.policy == "legacy":
        return LEGACY
    if args.policy == "codemod":
        from harness.codemod import codemod
        return ScriptedPolicy("codemod", lambda t: codemod(t["legacy"]))
    if args.policy.startswith("cheater:"):
        name = args.policy.split(":", 1)[1]
        return ScriptedPolicy(args.policy, CHEATERS[name])
    if args.policy == "openai":
        return OpenAIPolicy(args.model, args.base_url, args.api_key_env, args.temperature, args.max_tokens)
    raise SystemExit(f"unknown policy {args.policy!r}")


def hidden_seeds(spec):
    if spec == "train":
        return TRAIN_HIDDEN_SEEDS
    if spec.startswith("eval:"):
        return eval_seeds(spec.split(":", 1)[1])
    raise SystemExit("--seeds must be 'train' or 'eval:<round>'")


def summarize(episodes, tasks):
    by_id = {t["id"]: t for t in tasks}
    rewards = [e.reward for e in episodes]
    per_task, per_type = defaultdict(list), defaultdict(list)
    for e in episodes:
        passed = e.reward == 1.0
        per_task[e.task_id].append(passed)
        for ty in by_id[e.task_id]["types"]:
            per_type[ty].append(passed)
    rate = lambda xs: round(sum(xs) / len(xs), 4)
    return {
        "n_episodes": len(episodes),
        "pass_at_1": rate([r == 1.0 for r in rewards]),
        "mean_reward": round(sum(rewards) / len(rewards), 4),
        "reasons": dict(sorted(Counter(e.steps[-1].reason for e in episodes).items())),
        "per_task_pass": {k: rate(v) for k, v in sorted(per_task.items())},
        # tasks whose samples include both a pass and a failure: what gives GRPO a learning signal
        "mixed_task_frac": round(sum(0 < sum(v) < len(v) for v in per_task.values()) / max(len(per_task), 1), 4),
        "per_type_pass": {k: rate(v) for k, v in sorted(per_type.items())},
        "episode_rewards": [[e.task_id, e.sample_idx, e.reward] for e in episodes],
    }


def run(args):
    out = Path(args.out)
    tasks = load_tasks(args.tasks)[: args.limit] if args.limit else load_tasks(args.tasks)
    seeds = hidden_seeds(args.seeds)
    config = {k: v for k, v in vars(args).items() if k != "out"}
    record = manifest.build(config, tasks)
    if record["git"]["dirty"] and not args.allow_dirty:
        sys.exit("Refusing to run: uncommitted changes (or not a git repo). Commit first, or pass "
                 "--allow-dirty (the result is then marked non-reproducible).\n"
                 + "\n".join(record["git"]["dirty_files"]))
    out.mkdir(parents=True, exist_ok=True)
    limits, workers, t = Limits(), args.workers, {}

    # 1. inputs and legacy references (visible + hidden in one legacy job per task)
    t0 = time.monotonic()
    vis = {x["id"]: [make_inputs(x, s) for s in VISIBLE_SEEDS] for x in tasks}
    hid = {x["id"]: [make_inputs(x, s) for s in seeds] for x in tasks}
    with ThreadPoolExecutor(workers) as pool:
        refs = dict(zip([x["id"] for x in tasks],
                        pool.map(lambda x: compute_references(x, vis[x["id"]] + hid[x["id"]], limits), tasks)))
    nv = len(VISIBLE_SEEDS)
    fb = {}
    if args.feedback:                    # legacy code run on the VISIBLE inputs in the modern runtime
        from env.feedback import feedback
        with ThreadPoolExecutor(workers) as pool:
            fb = dict(zip([x["id"] for x in tasks], pool.map(
                lambda x: feedback(x, x["legacy"], vis[x["id"]], refs[x["id"]][:nv], limits)[0], tasks)))
    prompts = {x["id"]: prompt_mod.with_fewshot(prompt_mod.build(x, vis[x["id"]], refs[x["id"]][:nv], fb.get(x["id"])),
                                                x, args.fewshot)
               for x in tasks}
    t["references_s"] = time.monotonic() - t0

    # 2. completions: from the policy, or from stored trajectories when re-scoring
    t0 = time.monotonic()
    if args.rescore:
        stored = read_jsonl(args.rescore)
        policy_name = f"rescore:{stored[0].policy}" if stored else "rescore"
        completions = defaultdict(list)
        for e in sorted(stored, key=lambda e: (e.task_id, e.sample_idx)):
            completions[e.task_id].append(e.steps[-1].action)
    else:
        policy = make_policy(args)
        policy_name = policy.name + (f"+fewshot{args.fewshot}" if args.fewshot else "") + ("+feedback" if args.feedback else "")
        with ThreadPoolExecutor(args.gen_workers) as pool:
            gens = pool.map(lambda x: policy.sample(prompts[x["id"]], args.samples, x), tasks)
            completions = dict(zip([x["id"] for x in tasks], gens))
    t["generate_s"] = time.monotonic() - t0

    # 3. score every completion against the hidden references
    t0 = time.monotonic()
    jobs = [(x, i, c) for x in tasks for i, c in enumerate(completions[x["id"]])]
    with ThreadPoolExecutor(workers) as pool:
        rewards = list(pool.map(lambda j: score(j[0], j[2], hid[j[0]["id"]], refs[j[0]["id"]][nv:], limits), jobs))
    t["score_s"] = time.monotonic() - t0

    episodes = []
    for (x, i, c), r in zip(jobs, rewards):
        step = Step(prompt=prompts[x["id"]], action=c, reward=r.reward, reason=r.reason, detail=r.detail,
                    timing={"score_wall_s": round(r.wall_s, 3), "startup_s": round(r.startup_s, 3)})
        episodes.append(Episode(task_id=x["id"], sample_idx=i, policy=policy_name, hidden_seeds=seeds, steps=[step]))

    metrics = summarize(episodes, tasks)
    write_jsonl(out / "trajectories.jsonl", episodes)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))
    (out / "timing.json").write_text(json.dumps({k: round(v, 2) for k, v in t.items()}, indent=1))
    record["policy"] = policy_name
    record["result"] = {"pass_at_1": metrics["pass_at_1"], "reasons": metrics["reasons"]}
    (out / "run.json").write_text(json.dumps(record, indent=1))
    print(f"{policy_name}: pass@1 {metrics['pass_at_1']:.3f} on {metrics['n_episodes']} episodes; "
          f"mixed tasks {metrics['mixed_task_frac']:.3f}; "
          f"reasons {metrics['reasons']}; commit {(record['git']['commit'] or 'none')[:8]}"
          f"{' (DIRTY)' if record['git']['dirty'] else ''} -> {out}")
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="oracle", help="oracle | legacy | codemod | cheater:<name> | openai")
    p.add_argument("--rescore", help="trajectories.jsonl whose completions to re-score instead of sampling")
    p.add_argument("--tasks", default="seed", help="seed | train | heldout_id | heldout_cat | path/to/tasks.jsonl")
    p.add_argument("--limit", type=int, help="evaluate only the first N tasks (quick checks)")
    p.add_argument("--seeds", default="train", help="train | eval:<round>")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--feedback", action="store_true",
                   help="add the legacy code's errors/warnings on the visible examples to the prompt")
    p.add_argument("--fewshot", type=int, default=0, help="K hand-written demos as prior chat turns (baseline only)")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8, help="parallel sandbox jobs")
    p.add_argument("--gen-workers", type=int, default=4, help="parallel generation requests")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
