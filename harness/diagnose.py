"""Why did a run fail? Summarize one or more result directories written by harness.evaluate.

  envs/legacy/bin/python -m harness.diagnose results/calib_1p5b_train [results/base_1p5b_heldout_id ...]

For each run:
  - the error messages behind crash / warning / define_error, grouped (e.g. "no attribute 'append'")
  - per migration type: pass rate and the share of samples that crash
  - how often the model returned the legacy code unchanged, or did not keep the entry function
  - three example failing completions for the most common crash, to read by eye
"""
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from env.reward import extract_code
from harness.evaluate import load_tasks


def norm(code):
    try:
        return ast.unparse(ast.parse(code))
    except SyntaxError:
        return None


def error_key(detail):
    """Collapse an error message to its stable part: exception class + the API it names."""
    d = (detail or "").strip().splitlines()[-1] if detail else ""
    d = re.sub(r"'[^']{40,}'", "'...'", d)               # long quoted values
    d = re.sub(r"\b\d+(\.\d+)?\b", "N", d)                # numbers
    return d[:140]


def diagnose(run_dir):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "run.json"))["config"]
    tasks = load_tasks(cfg["tasks"])
    by_id = {t["id"]: t for t in tasks}
    eps = [json.loads(l) for l in open(run_dir / "trajectories.jsonl")]
    print(f"\n######## {run_dir}  ({len(eps)} episodes, policy {eps[0]['policy'] if eps else '?'})")

    errors = defaultdict(Counter)
    per_type = defaultdict(lambda: Counter())
    unchanged = renamed = 0
    examples = defaultdict(list)
    for e in eps:
        s = e["steps"][-1]
        t = by_id[e["task_id"]]
        code = extract_code(s["action"]) or ""
        n = norm(code)
        if n is not None and n == norm(t["legacy"]):
            unchanged += 1
        if n is not None and f"def {t['entry']}(" not in code:
            renamed += 1
        if s["reason"] in ("crash", "warning", "define_error", "timeout"):
            k = error_key(s["detail"])
            errors[s["reason"]][k] += 1
            if len(examples[k]) < 3:
                examples[k].append((e["task_id"], code))
        for ty in t["types"]:
            per_type[ty]["n"] += 1
            per_type[ty][s["reason"]] += 1

    print(f"returned the legacy code unchanged: {unchanged / len(eps):.1%}   "
          f"entry function missing/renamed: {renamed / len(eps):.1%}")
    for reason, c in errors.items():
        tot = sum(c.values())
        print(f"\n-- {reason}: {tot} ({tot / len(eps):.1%})")
        for k, v in c.most_common(12):
            print(f"  {v:5d}  {k}")
    print("\n-- per type: n  pass  crash  warning  mismatch")
    for ty, c in sorted(per_type.items(), key=lambda kv: kv[1]["pass"] / kv[1]["n"]):
        f = lambda r: f"{c[r] / c['n']:5.0%}"
        print(f"  {ty:28s} {c['n']:5d}  {f('pass')}  {f('crash')}  {f('warning')}  {f('mismatch')}")
    top = max((k for c in errors.values() for k in c), key=lambda k: sum(c[k] for c in errors.values()), default=None)
    if top:
        print(f"\n-- examples for the most common error: {top}")
        for tid, code in examples[top]:
            print(f"\n[{tid}]  types {by_id[tid]['types']}\n--- legacy:\n{by_id[tid]['legacy'].strip()}\n--- model:\n{code.strip()[:1500]}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        diagnose(d)
