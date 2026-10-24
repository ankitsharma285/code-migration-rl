"""Review 'migration: mismatch' rejections: is the comparator wrongly rejecting correct migrations?

Re-runs each rejected task (legacy code in legacy, Gemma's migration in modern) on the filter's inputs,
finds every place the two serialized outputs differ, and sorts each task into a category:

  values      the numbers/strings differ                  -> usually a genuinely wrong migration
  shape       lengths or shapes differ                    -> usually a genuinely wrong migration
  type        different kinds of object (Series vs DataFrame, list vs ndarray, ...)
  dtype       only dtype kinds differ                     -> possible comparator false negative
  names       only Series/index/column names differ       -> possible comparator false negative
  index       only index labels differ                    -> e.g. a reset or kept index
  mixed       several of the above

Prints a summary over ALL mismatch rejections, then N examples for manual judgment, favoring the
"possible false negative" categories. For each, decide: was the migration actually correct?

Usage (from the repo root, on a compute node):
  envs/legacy/bin/python -m tasks.review_mismatches tasks/generated/all_train/rejected.jsonl \
      tasks/generated/raw_train_s*.jsonl --n 15
"""
import argparse
import json
import math
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from env.compare import RTOL, ATOL, dtype_kind
from env.inputs import TRAIN_HIDDEN_SEEDS, VISIBLE_SEEDS, make_inputs
from env.runner import run_job
from tasks.filter_tasks import strip_comments

SEEDS = VISIBLE_SEEDS + TRAIN_HIDDEN_SEEDS
SUSPECT = ("dtype", "names", "index", "type")
NAME_KEYS = {"name", "index_names", "columns"}


def leaf_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def diffs(a, b, path=""):
    """List of (path, field, a_value, b_value) where the serialized outputs differ."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        if a.get("t") != b.get("t"):
            return [(path or "/", "type", a.get("t"), b.get("t"))]
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                out.append((p, "type", a.get(k, "<missing>"), b.get(k, "<missing>")))
            elif k in ("dtype", "dtypes"):
                av = a[k] if isinstance(a[k], list) else [a[k]]
                bv = b[k] if isinstance(b[k], list) else [b[k]]
                if [dtype_kind(x) for x in av] != [dtype_kind(x) for x in bv]:
                    out.append((p, "dtype", a[k], b[k]))
            elif k in NAME_KEYS:
                if a[k] != b[k]:
                    out.append((p, "names", a[k], b[k]))
            elif k == "index":
                if not (isinstance(a[k], list) and isinstance(b[k], list) and len(a[k]) == len(b[k])
                        and all(leaf_equal(x, y) for x, y in zip(a[k], b[k]))):
                    out.append((p, "index" if len(a[k]) == len(b[k]) else "shape", a[k], b[k]))
            elif k == "shape":
                if a[k] != b[k]:
                    out.append((p, "shape", a[k], b[k]))
            else:
                out += diffs(a[k], b[k], p)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(path, "shape", f"len {len(a)}", f"len {len(b)}")]
        for i, (x, y) in enumerate(zip(a, b)):
            out += diffs(x, y, f"{path}[{i}]")
            if len(out) > 20:
                break
        return out
    if isinstance(a, dict) != isinstance(b, dict) or isinstance(a, list) != isinstance(b, list):
        return [(path, "type", type(a).__name__, type(b).__name__)]
    return [] if leaf_equal(a, b) else [(path, "values", a, b)]


def categorize(found):
    fields = {f for _, f, _, _ in found}
    if not fields:
        return "no_diff"
    return fields.pop() if len(fields) == 1 else "mixed"


def review(rec):
    t = dict(rec["task"])
    t["legacy"], t["modern"] = strip_comments(t["legacy"]), strip_comments(t["modern"])
    inputs = [make_inputs(t, s) for s in SEEDS]
    leg = run_job("legacy", t["legacy"], t["entry"], inputs)
    mod = run_job("modern", t["modern"], t["entry"], inputs)
    if leg.status != "ok" or mod.status != "ok" or len(leg.calls) != len(mod.calls):
        return {"id": rec["request_id"], "category": "rerun_failed", "task": t}
    per_input = [diffs(l["result"], m["result"]) for l, m in zip(leg.calls, mod.calls)]
    bad = [(i, d) for i, d in enumerate(per_input) if d]
    if not bad:
        return {"id": rec["request_id"], "category": "no_diff", "task": t}
    cats = Counter(categorize(d) for _, d in bad)
    i, d = bad[0]
    return {"id": rec["request_id"], "task": t, "category": cats.most_common(1)[0][0] if len(cats) == 1 else "mixed",
            "inputs_differing": len(bad), "n_inputs": len(inputs), "first_input": inputs[i], "diffs": d[:8],
            "legacy_out": leg.calls[i]["result"], "modern_out": mod.calls[i]["result"],
            "types": rec.get("requested_types")}


def short(v, n=300):
    s = json.dumps(v)
    return s if len(s) <= n else s[:n] + " ..."


def show(r):
    print("=" * 100)
    print(f"{r['id']}  category={r['category']}  differs on {r.get('inputs_differing')}/{r.get('n_inputs')} inputs"
          f"  requested={r.get('types')}")
    print("-" * 46 + " LEGACY " + "-" * 46)
    print(r["task"]["legacy"].rstrip())
    print("-" * 44 + " MIGRATION " + "-" * 45)
    print(r["task"]["modern"].rstrip())
    if "diffs" not in r:
        return
    print("-" * 40 + " first differing input " + "-" * 37)
    print(short(r["first_input"], 500))
    print("-" * 46 + " DIFFERENCES (path: legacy -> migration) " + "-" * 13)
    for path, field, a, b in r["diffs"]:
        print(f"  [{field}] {path}: {short(a, 150)}  ->  {short(b, 150)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rejected", help="rejected.jsonl from tasks.filter_tasks")
    ap.add_argument("raw", nargs="+", help="the raw .jsonl files that run was built from")
    ap.add_argument("--n", type=int, default=15, help="examples to print")
    ap.add_argument("--only", help="print only this category")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    args = ap.parse_args()

    want = {json.loads(l)["request_id"] for l in open(args.rejected)
            if json.loads(l)["reason"].startswith("migration:") and "mismatch" in json.loads(l)["reason"]}
    recs = [r for path in args.raw for r in map(json.loads, open(path)) if r["request_id"] in want and r.get("task")]
    print(f"{len(recs)} 'migration: mismatch' rejections found; re-running them ...")
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(review, recs))

    counts = Counter(r["category"] for r in results)
    print("\nCategory counts over ALL mismatch rejections:")
    for c, k in counts.most_common():
        tag = "   <- possible comparator false negatives: review these" if c in SUSPECT else ""
        print(f"  {c:14s} {k:5d}{tag}")

    pool_ = [r for r in results if (r["category"] == args.only if args.only else "diffs" in r)]
    rng = random.Random(args.seed)
    suspects = [r for r in pool_ if r["category"] in SUSPECT]
    others = [r for r in pool_ if r["category"] not in SUSPECT]
    rng.shuffle(suspects), rng.shuffle(others)
    picks = (suspects[: (args.n + 1) // 2] + others)[: args.n] if not args.only else pool_[: args.n]
    print(f"\nShowing {len(picks)} examples (suspect categories first). For each: was the migration correct?\n")
    for r in picks:
        show(r)
    with open(os.path.splitext(args.rejected)[0] + "_mismatch_review.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "task"} | {"legacy": r["task"]["legacy"],
                                                                               "modern": r["task"]["modern"]},
                               default=str) + "\n")


if __name__ == "__main__":
    main()
