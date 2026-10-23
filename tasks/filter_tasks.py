"""Filter generated tasks. Nothing the generator says is trusted; every kept task is re-verified.

Checks, cheapest first (the first failure is the rejection reason):
  parse         the response had all four sections and a valid entry name
  ast           legacy and migration pass the AST check; the generator uses only random/math/builtins
  gen           the generator runs in the sandbox, is deterministic, and emits valid JSON inputs
  legacy        the legacy code runs on all 12 inputs (2 visible + 10 hidden) in the legacy runtime,
                deterministically, with outputs under the size cap
  types         the task's REAL migration types, detected from the code (static patterns) and from
                the errors/warnings the legacy code raises in the modern runtime; requested types
                are not trusted. Rejects held-out types (train split), unknown breakages, and none found.
  coverage      the legacy code fails, warns, or changes output in modern on >= 80% of the inputs
                (otherwise a wrong migration could still score 1 on the inputs that skip the old path)
  trap          no legacy output is an object column holding numbers (comparator rule)
  migration     the generator's migration scores 1 under the full reward (AST, no warnings, match)
  duplicate     not a structural duplicate of an already-kept task (normalized AST of the legacy code)

Usage (from the repo root):
  envs/legacy/bin/python -m tasks.filter_tasks tasks/generated/raw_pilot.jsonl --out tasks/generated/pilot
Writes <out>/accepted.jsonl, <out>/rejected.jsonl, <out>/stats.json.
"""
import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from env.ast_check import check as ast_violations
from env.compare import matches
from env.inputs import TRAIN_HIDDEN_SEEDS, VISIBLE_SEEDS, make_inputs
from env.reward import score
from env.runner import run_job
from harness.policies import fence
from tasks.check_seeds import object_holding_numbers

HERE = Path(__file__).resolve().parent
CATS = json.loads((HERE / "categories.json").read_text())
TRAIN = set(CATS["train"]["behavior"]) | set(CATS["train"]["rename"])
BEHAVIOR = set(CATS["train"]["behavior"])
HELD_OUT = set(CATS["held_out"])
SEEDS = VISIBLE_SEEDS + TRAIN_HIDDEN_SEEDS
MIN_COVERAGE = 0.8
MAX_OUTPUT_CHARS = 20_000

# Static patterns in the LEGACY code: unambiguous signatures of a migration type.
STATIC = {
    "iteritems": r"\.iteritems\(",
    "applymap": r"\.applymap\(",
    "level_reduction": r"\.(sum|mean|max|min|count|std|var|prod|median|any|all|sem|skew)\([^)]*\blevel\s*=",
    "int64index": r"\b(Int64Index|UInt64Index|Float64Index)\b",
    "to_csv_line_terminator": r"\bline_terminator\s*=",
    "read_csv_squeeze": r"\bsqueeze\s*=\s*True",
    "str_replace_regex": r"\.str\.replace\((?![^)]*\bregex\s*=)",
    "fillna_method": r"\.fillna\([^)]*\bmethod\s*=",
    "np_type_aliases": r"\bnp\.(float|int|bool|object|str|complex|long|unicode)\b(?!\d)",
    "np_product_family": r"\bnp\.(product|cumproduct|alltrue|sometrue)\(",
    "np_trapz": r"\bnp\.trapz\(",
    "np_nan_inf_aliases": r"\bnp\.(NaN|Inf|PINF|NINF|Infinity|infty|NAN)\b",
    "np_in1d": r"\bnp\.in1d\(",
    "np_row_stack": r"\bnp\.row_stack\(",
    "ndarray_ptp": r"(?<!np)\.ptp\(",
    "find_common_type": r"\bnp\.find_common_type\(",
    "array_copy_false": r"\bnp\.array\([^)]*\bcopy\s*=\s*False",
    "mad": r"\.mad\(",
    "dt_weekofyear": r"\.(weekofyear|dt\.week)\b",
    "lookup": r"\.lookup\(",
}
# Messages the legacy code raises when run in the MODERN runtime.
RUNTIME = {
    "df_append_loop": r"'DataFrame' object has no attribute 'append'",
    "series_append": r"'Series' object has no attribute 'append'",
    "groupby_numeric_only": r"agg function failed|could not convert string to float|unsupported operand type.*'str'",
    "groupby_observed": r"default of observed=False",
    "series_positional_getitem": r"treating keys as positions",
    "groupby_apply_groups": r"operated on the grouping columns",
    "chained_inplace": r"chained assignment using an inplace method",
    "iteritems": r"no attribute 'iteritems'",
    "applymap": r"applymap has been deprecated",
    "int64index": r"no attribute '(Int64Index|UInt64Index|Float64Index)'",
    "level_reduction": r"unexpected keyword argument 'level'",
    "read_csv_squeeze": r"unexpected keyword argument 'squeeze'",
    "to_csv_line_terminator": r"unexpected keyword argument 'line_terminator'",
    "fillna_method": r"fillna with 'method' is deprecated",
    "np_type_aliases": r"has no attribute '(float|int|bool|object|str|complex)'",
    "np_product_family": r"has no attribute '(product|cumproduct|alltrue|sometrue)'",
    "np_trapz": r"`trapz` is deprecated",
    "np_nan_inf_aliases": r"`np\.(NaN|Inf|PINF|NINF|Infinity|infty)` was removed",
    "np_in1d": r"`in1d` is deprecated",
    "np_row_stack": r"`row_stack` alias is deprecated",
    "ndarray_ptp": r"`ptp` was removed from the ndarray",
    "find_common_type": r"find_common_type` was removed",
    "array_copy_false": r"Unable to avoid copy",
    "mad": r"no attribute 'mad'",
    "dt_weekofyear": r"no attribute '(weekofyear|week)'",
    "lookup": r"no attribute 'lookup'",
}
GEN_ALLOWED_IMPORTS = {"random", "math", "string", "itertools", "datetime"}


class Reject(Exception):
    pass


def check_gen_source(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise Reject(f"gen: syntax error line {e.lineno}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(n.split(".")[0] not in GEN_ALLOWED_IMPORTS for n in names):
                raise Reject(f"gen: import not allowed: {names}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise Reject("gen: private attribute")
        if isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "__import__", "getattr", "compile"}:
            raise Reject(f"gen: uses {node.id}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "gen" for n in tree.body):
        raise Reject("gen: no function named gen")


def unser(x):
    """Inverse of the sandbox serializer for plain JSON-like data (generator outputs)."""
    t = x["t"]
    if t == "dict":
        return {k: unser(v) for k, v in x["items"].items()}
    if t == "list":
        return [unser(v) for v in x["items"]]
    if t in ("str", "int", "bool", "NoneType"):
        return x["value"]
    if t == "float":
        if isinstance(x["value"], str):
            raise Reject("gen: produced NaN/inf; use None for missing values")
        return x["value"]
    raise Reject(f"gen: produced a non-JSON value of type {t}")


def normalized_hash(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):                      # ignore names and literals: structure only
        if isinstance(node, ast.Name):
            node.id = "_"
        elif isinstance(node, ast.arg):
            node.arg = "_"
        elif isinstance(node, ast.FunctionDef):
            node.name = "_"
        elif isinstance(node, ast.Constant):
            node.value = 0
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def detect_types(legacy, messages):
    found = {t for t, pat in STATIC.items() if re.search(pat, legacy)}
    unknown = []
    for m in messages:
        hits = {t for t, pat in RUNTIME.items() if re.search(pat, m)}
        found |= hits
        if not hits:
            unknown.append(m)
    return found, unknown


def verify(rec, split):
    task = rec.get("task")
    if not task:
        raise Reject(f"parse: {rec.get('parse_error') or 'unparseable'}")
    task = {**task, "id": rec["request_id"]}
    for name in ("legacy", "modern"):
        try:
            ast.parse(task[name])
        except SyntaxError as e:
            raise Reject(f"ast: {name} syntax error line {e.lineno}")
        v = ast_violations(task[name])
        if v:
            raise Reject(f"ast: {name}: {v[0]}")
    check_gen_source(task["gen"])

    # generator: run sandboxed first (untrusted code), then compare with the parent-side generation
    res = run_job("legacy", task["gen"] + "\nimport random\ndef make_inputs_for(seed):\n"
                  "    return gen(random.Random(seed))\n", "make_inputs_for", [{"seed": s} for s in SEEDS])
    if res.status != "ok" or res.define.get("status") == "FAIL" or any(c["status"] == "FAIL" for c in res.calls):
        bad = next((c["detail"] for c in res.calls if c["status"] == "FAIL"), res.detail or res.define.get("detail"))
        raise Reject(f"gen: failed in sandbox: {bad}"[:200])
    sandbox_inputs = [unser(c["result"]) for c in res.calls]
    if not all(isinstance(i, dict) and i for i in sandbox_inputs):
        raise Reject("gen: must return a non-empty dict of keyword arguments")
    inputs = [make_inputs(task, s) for s in SEEDS]
    if inputs != sandbox_inputs:
        raise Reject("gen: not deterministic for a given seed")

    # legacy behavior
    leg = run_job("legacy", task["legacy"], task["entry"], inputs)
    leg2 = run_job("legacy", task["legacy"], task["entry"], inputs)
    if leg.status != "ok" or leg.define.get("status") == "FAIL":
        raise Reject(f"legacy: {leg.status} {leg.detail or leg.define.get('detail')}"[:200])
    fails = [c for c in leg.calls if c["status"] == "FAIL"]
    if fails:
        raise Reject(f"legacy: fails on an input: {fails[0]['detail']}"[:200])
    refs = [c["result"] for c in leg.calls]
    if leg2.status != "ok" or [c.get("result") for c in leg2.calls] != refs:
        raise Reject("legacy: not deterministic")
    if any(len(json.dumps(r)) > MAX_OUTPUT_CHARS for r in refs):
        raise Reject("legacy: output too large for the prompt")

    # what the legacy code does in the modern runtime
    mod = run_job("modern", task["legacy"], task["entry"], inputs)
    messages, broken = [], 0
    if mod.status != "ok":
        raise Reject(f"coverage: legacy in modern {mod.status}: {mod.detail}"[:200])
    if mod.define.get("status") == "FAIL":
        messages.append(mod.define["detail"])
        broken = len(inputs)
    else:
        if mod.define.get("status") == "WARN":
            messages.append(mod.define["warning"])
        for c, ref in zip(mod.calls, refs):
            if c["status"] == "FAIL":
                messages.append(c["detail"])
                broken += 1
            elif c["status"] == "WARN":
                messages.append(c["warning"])
                broken += 1
            elif not matches(ref, c["result"]):
                messages.append("SILENT_CHANGE")
                broken += 1
    silent = "SILENT_CHANGE" in messages
    messages = [m for m in messages if m != "SILENT_CHANGE"]
    types, unknown = detect_types(task["legacy"], messages)
    if silent and "str_replace_regex" not in types:   # the only verified silent-change type
        raise Reject("types: output changes silently in modern and no known type explains it")
    if unknown:
        raise Reject(f"types: unknown breakage: {unknown[0][:150]}")
    if not types:
        raise Reject("types: no known migration type found in the code")
    if split == "train" and types & HELD_OUT:
        raise Reject(f"types: uses held-out type(s) {sorted(types & HELD_OUT)}")
    if split == "heldout" and not types & HELD_OUT:
        raise Reject("types: held-out task without a held-out type")
    coverage = broken / len(inputs)
    if coverage < MIN_COVERAGE:
        raise Reject(f"coverage: old code path exercised on only {coverage:.0%} of inputs")
    if any(object_holding_numbers(r) for r in refs):
        raise Reject("trap: legacy output has object dtype holding numbers")

    nv = len(VISIBLE_SEEDS)
    r = score(task, fence(task["modern"]), inputs[nv:], refs[nv:])
    if r.reward != 1.0:
        raise Reject(f"migration: generator's migration scores 0 ({r.reason}: {r.detail})"[:200])

    return {"id": task["id"], "types": sorted(types), "entry": task["entry"], "legacy": task["legacy"],
            "modern": task["modern"], "gen": task["gen"], "split": split, "domain": rec.get("domain"),
            "requested_types": rec.get("requested_types"), "model": rec.get("model"),
            "coverage": round(coverage, 2), "behavior": bool(types & BEHAVIOR)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="+", help="raw .jsonl files from tasks.generate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    args = ap.parse_args()

    recs = [json.loads(line) for path in args.raw for line in open(path)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def one(rec):
        try:
            return verify(rec, rec.get("split", "train")), None
        except Reject as e:
            return None, str(e)
        except Exception as e:                         # a bug in this script, not in the task
            return None, f"internal: {type(e).__name__}: {e}"[:200]

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, recs))

    accepted, rejected, seen = [], [], set()
    for rec, (task, why) in zip(recs, results):
        if task:
            h = normalized_hash(task["legacy"])
            if h in seen:
                task, why = None, "duplicate: structurally identical to a kept task"
            else:
                seen.add(h)
        if task:
            accepted.append(task)
        else:
            rejected.append({"request_id": rec["request_id"], "reason": why,
                             "requested_types": rec.get("requested_types")})

    with open(out / "accepted.jsonl", "w") as f:
        for t in accepted:
            f.write(json.dumps(t) + "\n")
    with open(out / "rejected.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")

    n = len(recs)
    stage = Counter(r["reason"].split(":")[0] for r in rejected)
    type_counts = Counter(t for a in accepted for t in a["types"])
    asked = sum(len(set(a["requested_types"] or [])) for a in accepted)
    honored = sum(len(set(a["requested_types"] or []) & set(a["types"])) for a in accepted)
    stats = {
        "requests": n, "accepted": len(accepted), "yield": round(len(accepted) / max(n, 1), 3),
        "rejected_by_stage": dict(stage.most_common()),
        "behavior_share_of_accepted": round(sum(a["behavior"] for a in accepted) / max(len(accepted), 1), 3),
        "requested_types_honored": round(honored / max(asked, 1), 3),
        "accepted_per_type": dict(sorted(type_counts.items())),
        "types_never_accepted": sorted((TRAIN if all(r.get("split", "train") == "train" for r in recs) else HELD_OUT)
                                       - set(type_counts)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"\ntop rejection reasons:")
    for reason, k in Counter(r["reason"][:90] for r in rejected).most_common(8):
        print(f"  {k:4d}  {reason}")


if __name__ == "__main__":
    main()
