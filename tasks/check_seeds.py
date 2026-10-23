"""Check every seed task against the pinned runtimes and the comparator rules.

For each task and each input seed (2 visible + 10 hidden):
  legacy code  in legacy env  -> must run (deprecation warnings allowed)
  legacy code  in modern env  -> must fail, warn, or differ on at least one input
  modern code  in modern env  -> must run with no deprecation warnings and match legacy
Plus: legacy is deterministic (same input twice gives the same output), the task uses only
training types, the source mentions no held-out API, and no legacy output is an object
column holding numbers (filtered by the comparator rules).

Usage (from the repo root):
    envs/legacy/bin/python tasks/check_seeds.py
Writes tasks/seed_report.md.
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seed_tasks import SEED_TASKS  # noqa: E402
from verify_types import run_env, same  # noqa: E402

VISIBLE_SEEDS = [0, 1]
HIDDEN_SEEDS = list(range(100, 110))
HELD_OUT_PATTERNS = [r"\.mad\(", r"weekofyear", r"\.lookup\("]

SETUP = '''import io, json
import numpy as np
import pandas as pd
def _mat(v):
    if isinstance(v, dict) and "__df__" in v:
        return pd.DataFrame(v["__df__"])
    if isinstance(v, dict) and "__series__" in v:
        d = v["__series__"]
        idx = d.get("index")
        if idx and isinstance(idx[0], list):
            idx = pd.MultiIndex.from_tuples([tuple(i) for i in idx])
        return pd.Series(d["values"], index=idx, name=d.get("name"))
    return v
kwargs = {k: _mat(v) for k, v in json.loads(%r).items()}
'''


def make_inputs(task, seed):
    ns = {}
    exec(task["gen"], ns)
    return ns["gen"](random.Random(seed))


def object_holding_numbers(res):
    """True if a serialized result has an object-dtype column/series whose values are all numbers."""
    def numeric(vals):
        flat = vals if not vals or not isinstance(vals[0], list) else [x for row in vals for x in row]
        return bool(flat) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in flat)
    if not isinstance(res, dict):
        return False
    t = res.get("t")
    if t == "series":
        return res["dtype"] == "object" and numeric(res["values"])
    if t == "df":
        for j, dt in enumerate(res["dtypes"]):
            if dt == "object" and numeric([row[j] for row in res["values"]]):
                return True
        return False
    if t == "list":
        return any(object_holding_numbers(i) for i in res["items"])
    if t == "dict":
        return any(object_holding_numbers(i) for i in res["items"].values())
    return False


def main():
    cats = json.loads((HERE / "categories.json").read_text())
    train = set(cats["train"]["behavior"]) | set(cats["train"]["rename"])
    held = set(cats["held_out"])
    envs = HERE.parent / "envs"

    legacy_specs, modern_specs = [], []
    for t in SEED_TASKS:
        for seed in VISIBLE_SEEDS + HIDDEN_SEEDS:
            setup = SETUP % json.dumps(make_inputs(t, seed))
            call = f"\nresult = {t['entry']}(**kwargs)\n"
            k = f"{t['id']}|{seed}"
            legacy_specs.append(dict(key=k, setup=setup, code=t["legacy"] + call))
            modern_specs.append(dict(key=k + "|orig", setup=setup, code=t["legacy"] + call))
            modern_specs.append(dict(key=k + "|rewrite", setup=setup, code=t["modern"] + call))
        rerun_setup = SETUP % json.dumps(make_inputs(t, HIDDEN_SEEDS[0]))
        legacy_specs.append(dict(key=f"{t['id']}|rerun", setup=rerun_setup,
                                 code=t["legacy"] + f"\nresult = {t['entry']}(**kwargs)\n"))

    L = run_env(str(envs / "legacy" / "bin" / "python"), legacy_specs)
    M = run_env(str(envs / "modern" / "bin" / "python"), modern_specs)

    rows, all_ok = [], True
    coverage = {}
    for t in SEED_TASKS:
        problems = []
        bad_types = [x for x in t["types"] if x not in train]
        if bad_types:
            problems.append(f"not training types: {bad_types}")
        if set(t["types"]) & held or any(re.search(p, t["legacy"] + t["modern"]) for p in HELD_OUT_PATTERNS):
            problems.append("uses a held-out type")
        for x in t["types"]:
            coverage[x] = coverage.get(x, 0) + 1

        needs, legacy_warns = False, False
        for seed in VISIBLE_SEEDS + HIDDEN_SEEDS:
            k = f"{t['id']}|{seed}"
            lg, orig, rw = L[k], M[k + "|orig"], M[k + "|rewrite"]
            if lg["status"] == "FAIL":
                problems.append(f"seed {seed}: legacy fails: {lg['detail']}")
                continue
            legacy_warns |= lg["status"] == "WARN"
            if object_holding_numbers(lg["result"]):
                problems.append(f"seed {seed}: legacy output has object dtype holding numbers")
            if orig["status"] in ("FAIL", "WARN") or not same(lg["result"], orig["result"], "kind"):
                needs = True
            if rw["status"] != "OK":
                problems.append(f"seed {seed}: rewrite {rw['status']}: {rw['detail']}")
            elif not same(lg["result"], rw["result"], "kind"):
                problems.append(f"seed {seed}: rewrite output differs from legacy")
        if not needs:
            problems.append("legacy code needs no migration on any input")
        rerun = L[f"{t['id']}|rerun"]
        first = L[f"{t['id']}|{HIDDEN_SEEDS[0]}"]
        if rerun.get("result") != first.get("result"):
            problems.append("legacy output not deterministic")

        ok = not problems
        all_ok &= ok
        orig_status = sorted({M[f"{t['id']}|{s}|orig"]["status"] for s in VISIBLE_SEEDS + HIDDEN_SEEDS})
        rows.append(dict(id=t["id"], types=", ".join(t["types"]), ok=ok,
                         legacy="WARN" if legacy_warns else "OK",
                         modern="/".join(orig_status), problems=problems))

    missing = sorted(train - set(coverage))
    lines = ["# Seed task check", "",
             f"{sum(r['ok'] for r in rows)}/{len(rows)} tasks pass on "
             f"{len(VISIBLE_SEEDS) + len(HIDDEN_SEEDS)} inputs each. "
             f"Training types covered: {len(coverage)}/{len(train)}"
             + (f" (missing: {', '.join(missing)})" if missing else "") + ".", "",
             "| task | types | legacy | legacy code in modern | pass |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['types']} | {r['legacy']} | {r['modern']} | "
                     f"{'yes' if r['ok'] else 'NO'} |")
    failures = [r for r in rows if r["problems"]]
    if failures:
        lines += ["", "## Problems", ""]
        for r in failures:
            for p in dict.fromkeys(r["problems"]):
                lines.append(f"- **{r['id']}**: {p}")
    (HERE / "seed_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if all_ok and not missing else 1)


if __name__ == "__main__":
    main()
