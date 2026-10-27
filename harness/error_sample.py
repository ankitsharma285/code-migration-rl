"""Stage 7 error analysis: a random sample of failed episodes, written out for reading and bucketing.

    python -m harness.error_sample results/final/tp_ck00150_mt3_heldout_id,results/final/tp2_ck00150_mt3_heldout_id \\
        --n 50 --out results/final/errors_tp_heldout_id.md

Samples failed episodes uniformly across the given runs (one episode per task at most, so a few hard tasks
cannot fill the sample), and prints for each: task, migration types, failure reason and message, turns used,
the feedback before the last attempt, the legacy code, and the final code. Counts by reason come first.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from env.reward import extract_code
from harness.diagnose import error_key
from harness.evaluate import load_tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = args.runs.split(",")
    cfg = json.load(open(Path(dirs[0]) / "run.json"))["config"]
    tasks = {t["id"]: t for t in load_tasks(cfg["tasks"])}
    fails = []
    for d in dirs:
        for l in open(Path(d) / "trajectories.jsonl"):
            e = json.loads(l)
            if e["steps"] and e["steps"][-1]["reason"] != "pass":
                fails.append((d, e))
    rng = random.Random(args.seed)
    rng.shuffle(fails)
    sample, seen = [], set()
    for d, e in fails:
        if e["task_id"] not in seen:
            seen.add(e["task_id"])
            sample.append((d, e))
        if len(sample) == args.n:
            break
    n_eps = sum(1 for d in dirs for _ in open(Path(d) / "trajectories.jsonl"))
    lines = [f"# Error sample: {args.runs}", "",
             f"{len(fails)} failed episodes of {n_eps}; {len({e['task_id'] for _, e in fails})} tasks with a failure;"
             f" sample of {len(sample)} (one per task).", "",
             "Reasons in the sample: " + ", ".join(f"{k} {v}" for k, v in Counter(e["steps"][-1]["reason"] for _, e in sample).most_common()),
             "", "Messages in the sample: ",
             *[f"- {v} x `{k}`" for k, v in Counter(error_key(e["steps"][-1]["detail"]) for _, e in sample).most_common(15)], ""]
    for i, (d, e) in enumerate(sample, 1):
        t, last = tasks[e["task_id"]], e["steps"][-1]
        fb = e["steps"][-2]["tool_output"] if len(e["steps"]) > 1 and e["steps"][-2].get("tool_output") else ""
        lines += [f"## {i}. {e['task_id']} ({', '.join(t['types'])})", "",
                  f"reason **{last['reason']}**: `{(last['detail'] or '').splitlines()[0][:200] if last['detail'] else ''}`  ",
                  f"turns {len(e['steps'])}, stop {e.get('info', {}).get('stop', '-')}, run {Path(d).name}", "",
                  "Bucket: ", ""]
        if fb:
            lines += ["Feedback before the last attempt:", "```text", fb[:800], "```"]
        lines += ["Legacy:", "```python", t["legacy"].strip(), "```", "Final code:", "```python",
                  (extract_code(last["action"]) or last["action"] or "").strip()[:2500], "```", ""]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}: {len(sample)} failures")


if __name__ == "__main__":
    main()
