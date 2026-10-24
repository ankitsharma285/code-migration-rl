"""Verify candidate migration types against the pinned legacy and modern runtimes.

For each probe it runs:
  1. the legacy code in the legacy env        -> must run (warnings allowed)
  2. the same legacy code in the modern env   -> fails or warns = "detectable";
                                                 runs but different output = "silent"
  3. the hand-written rewrite in the modern env -> must run with no deprecation warnings
and compares the rewrite's output with the legacy output (values, then dtypes).

Usage (from the repo root):
    envs/legacy/bin/python tasks/verify_types.py
Writes tasks/migration_types.md and tasks/migration_types.json.
History notes verified 2026-10-22 against pandas/NumPy release notes and test installs of pandas 2.0.3 and 2.1.0.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PD = "import io\nimport numpy as np\nimport pandas as pd\n"
NP = "import numpy as np\n"

# kind: rename = mechanical API swap; behavior = needs restructuring or a semantic decision;
#       trap = not a migration type, a comparator hazard (legacy and modern code identical)
PROBES = [
    # ---------------- pandas ----------------
    dict(id="df_append_loop", kind="behavior", lib="pandas",
         history="DataFrame.append: deprecated 1.4, removed 2.0",
         setup=PD,
         legacy="out = pd.DataFrame({'a': [0], 'b': [0]})\n"
                "for i in range(1, 4):\n"
                "    out = out.append({'a': i, 'b': i * 2}, ignore_index=True)\n"
                "result = out",
         modern="rows = [{'a': 0, 'b': 0}] + [{'a': i, 'b': i * 2} for i in range(1, 4)]\n"
                "result = pd.DataFrame(rows)"),
    dict(id="series_append", kind="rename", lib="pandas",
         history="Series.append: deprecated 1.4, removed 2.0",
         setup=PD,
         legacy="result = pd.Series([1, 2]).append(pd.Series([3]), ignore_index=True)",
         modern="result = pd.concat([pd.Series([1, 2]), pd.Series([3])], ignore_index=True)"),
    dict(id="iteritems", kind="rename", lib="pandas",
         history="iteritems: deprecated 1.5, removed 2.0",
         setup=PD + "df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})\n",
         legacy="result = {k: v.sum() for k, v in df.iteritems()}",
         modern="result = {k: v.sum() for k, v in df.items()}"),
    dict(id="applymap", kind="rename", lib="pandas",
         history="DataFrame.applymap: deprecated 2.1 (renamed DataFrame.map)",
         setup=PD + "df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})\n",
         legacy="result = df.applymap(lambda v: v * 2)",
         modern="result = df.map(lambda v: v * 2)"),
    dict(id="groupby_numeric_only", kind="behavior", lib="pandas",
         history="numeric_only=None deprecated 1.5; 2.0 default numeric_only=False raises on non-numeric columns",
         setup=PD + "df = pd.DataFrame({'k': ['a', 'a', 'b'], 'v': [1, 2, 3], 's': ['x', 'y', 'z']})\n",
         legacy="result = df.groupby('k').mean()",
         modern="result = df.groupby('k').mean(numeric_only=True)"),
    dict(id="lookup", kind="behavior", lib="pandas",
         history="DataFrame.lookup: deprecated 1.2, removed 2.0",
         setup=PD + "df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})\n",
         legacy="result = df.lookup([0, 1, 2], ['x', 'y', 'x'])",
         modern="result = df.to_numpy()[df.index.get_indexer([0, 1, 2]), df.columns.get_indexer(['x', 'y', 'x'])]"),
    dict(id="mad", kind="behavior", lib="pandas",
         history="DataFrame.mad: deprecated 1.5, removed 2.0",
         setup=PD + "df = pd.DataFrame({'x': [1.0, 2.0, 6.0], 'y': [4.0, 4.0, 7.0]})\n",
         legacy="result = df.mad()",
         modern="result = (df - df.mean()).abs().mean()"),
    dict(id="level_reduction", kind="behavior", lib="pandas",
         history="level= in reductions: deprecated 1.3, removed 2.0",
         # UNSORTED index on purpose: sum(level=) keeps first-appearance order, groupby sorts by default,
         # so the textbook groupby(level=0).sum() is only equivalent on already-sorted input.
         setup=PD + "s = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples("
                    "[('b', 1), ('b', 2), ('a', 1), ('a', 2)]))\n",
         legacy="result = s.sum(level=0)",
         modern="result = s.groupby(level=0, sort=False).sum()"),
    dict(id="int64index", kind="rename", lib="pandas",
         history="Int64Index/Float64Index: deprecated 1.4, removed 2.0",
         setup=PD,
         legacy="result = pd.Int64Index([1, 2, 3])",
         modern="result = pd.Index([1, 2, 3], dtype='int64')"),
    dict(id="to_csv_line_terminator", kind="rename", lib="pandas",
         history="to_csv(line_terminator=): deprecated 1.5, removed 2.0",
         setup=PD + "df = pd.DataFrame({'x': [1, 2]})\nbuf = io.StringIO()\n",
         legacy="df.to_csv(buf, line_terminator='\\n')\nresult = buf.getvalue()",
         modern="df.to_csv(buf, lineterminator='\\n')\nresult = buf.getvalue()"),
    dict(id="read_csv_squeeze", kind="behavior", lib="pandas",
         history="read_csv(squeeze=): deprecated 1.4, removed 2.0",
         setup=PD,
         legacy="result = pd.read_csv(io.StringIO('a\\n1\\n2\\n'), squeeze=True)",
         modern="result = pd.read_csv(io.StringIO('a\\n1\\n2\\n')).squeeze('columns')"),
    dict(id="str_replace_regex", kind="behavior", lib="pandas",
         history="str.replace: regex default change announced 1.2, became regex=False in 2.0; single-char patterns already literal in 1.x",
         setup=PD + "s = pd.Series(['a.b', 'c.d'])\n",
         legacy="result = s.str.replace('[.]', '-')",
         modern="result = s.str.replace('[.]', '-', regex=True)"),
    dict(id="groupby_observed", kind="behavior", lib="pandas",
         history="groupby observed=False default: deprecated 2.1",
         setup=PD + "df = pd.DataFrame({'k': pd.Categorical(['a', 'a'], categories=['a', 'b']), 'v': [1, 2]})\n",
         legacy="result = df.groupby('k')['v'].sum()",
         modern="result = df.groupby('k', observed=False)['v'].sum()"),
    dict(id="dt_weekofyear", kind="behavior", lib="pandas",
         history="Series.dt.weekofyear: deprecated 1.1, removed 2.0 (isocalendar().week is UInt32 and named 'week')",
         setup=PD + "s = pd.Series(pd.to_datetime(['2021-01-04', '2021-06-15', '2021-12-31']))\n",
         legacy="result = s.dt.weekofyear",
         modern="result = s.dt.isocalendar().week.astype('int64').rename(None)"),
    dict(id="series_positional_getitem", kind="behavior", lib="pandas",
         history="Series[int] on a non-integer index as position: deprecated 2.1 (choose .iloc or .loc)",
         setup=PD + "s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])\n",
         legacy="result = [s[0], s[-1]]",
         modern="result = [s.iloc[0], s.iloc[-1]]"),
    dict(id="groupby_apply_groups", kind="behavior", lib="pandas",
         history="groupby.apply including grouping columns: deprecated 2.2 (include_groups=False)",
         setup=PD + "df = pd.DataFrame({'k': ['a', 'a', 'b'], 'v': [1, 2, 4]})\n",
         legacy="result = df.groupby('k').apply(lambda g: g['v'].sum() / len(g))",
         modern="result = df.groupby('k').apply(lambda g: g['v'].sum() / len(g), include_groups=False)"),
    dict(id="chained_inplace", kind="behavior", lib="pandas",
         history="chained assignment with inplace=True: warns in 2.2, stops working under copy-on-write (3.0)",
         setup=PD + "df = pd.DataFrame({'a': [1.0, None, 3.0], 'b': [1, 2, 3]})\n",
         legacy="df['a'].fillna(0, inplace=True)\nresult = df",
         modern="df['a'] = df['a'].fillna(0)\nresult = df"),
    dict(id="fillna_method", kind="rename", lib="pandas",
         history="fillna(method=): deprecated 2.1",
         setup=PD + "s = pd.Series([1.0, None, 3.0])\n",
         legacy="result = s.fillna(method='ffill')",
         modern="result = s.ffill()"),
    # ---------------- numpy ----------------
    dict(id="np_type_aliases", kind="rename", lib="numpy",
         history="np.float/np.int/...: deprecated 1.20, removed 1.24",
         setup=NP,
         legacy="result = np.array([1.5, 2.5], dtype=np.float).sum() + np.int(1)",
         modern="result = np.array([1.5, 2.5], dtype=float).sum() + int(1)"),
    dict(id="np_product_family", kind="rename", lib="numpy",
         history="product/cumproduct/alltrue/sometrue: deprecated 1.25, removed 2.0",
         setup=NP,
         legacy="result = [np.product([1, 2, 3]), np.cumproduct([1, 2, 3]),"
                " np.alltrue([True, False]), np.sometrue([True, False])]",
         modern="result = [np.prod([1, 2, 3]), np.cumprod([1, 2, 3]),"
                " np.all([True, False]), np.any([True, False])]"),
    dict(id="np_trapz", kind="rename", lib="numpy",
         history="np.trapz: deprecated 2.0 (renamed np.trapezoid)",
         setup=NP,
         legacy="result = np.trapz([1.0, 2.0, 4.0])",
         modern="result = np.trapezoid([1.0, 2.0, 4.0])"),
    dict(id="np_nan_inf_aliases", kind="rename", lib="numpy",
         history="np.NaN/np.Inf/np.PINF etc.: removed 2.0",
         setup=NP,
         legacy="result = np.array([np.NaN, np.Inf, 1.0])",
         modern="result = np.array([np.nan, np.inf, 1.0])"),
    dict(id="np_in1d", kind="rename", lib="numpy",
         history="np.in1d: deprecated 2.0 (use np.isin)",
         setup=NP,
         legacy="result = np.in1d([1, 2, 3], [2])",
         modern="result = np.isin([1, 2, 3], [2])"),
    dict(id="np_row_stack", kind="rename", lib="numpy",
         history="np.row_stack: deprecated 2.0 (use np.vstack)",
         setup=NP,
         legacy="result = np.row_stack([[1, 2], [3, 4]])",
         modern="result = np.vstack([[1, 2], [3, 4]])"),
    dict(id="ndarray_ptp", kind="rename", lib="numpy",
         history="ndarray.ptp method: removed 2.0 (np.ptp remains)",
         setup=NP,
         legacy="result = np.arange(5).ptp()",
         modern="result = np.ptp(np.arange(5))"),
    dict(id="find_common_type", kind="behavior", lib="numpy",
         history="np.find_common_type: deprecated 1.25, removed 2.0",
         setup=NP,
         legacy="result = np.find_common_type([np.int32, np.float32], [])",
         modern="result = np.result_type(np.int32, np.float32)"),
    dict(id="array_copy_false", kind="behavior", lib="numpy",
         history="np.array(copy=False) raises in 2.0 when a copy is needed",
         setup=NP,
         legacy="result = np.array([1, 2, 3], copy=False)",
         modern="result = np.asarray([1, 2, 3])"),
    # ---------------- comparator traps (same code both sides) ----------------
    dict(id="trap_nep50_scalar", kind="trap", lib="numpy",
         history="NEP 50 (NumPy 2.0): Python scalars no longer upcast float32",
         setup=NP,
         legacy="result = np.float32(3) + 3.0",
         modern="result = np.float32(3) + 3.0"),
    dict(id="trap_value_counts_names", kind="trap", lib="pandas",
         history="value_counts result/index names changed in 2.0",
         setup=PD + "df = pd.DataFrame({'a': ['x', 'y', 'x']})\n",
         legacy="result = df['a'].value_counts()",
         modern="result = df['a'].value_counts()"),
    dict(id="trap_append_empty_dtype", kind="trap", lib="pandas",
         history="appending dicts to an empty frame yields object dtype in 1.x",
         setup=PD,
         legacy="out = pd.DataFrame(columns=['a', 'b'])\n"
                "for i in range(3):\n"
                "    out = out.append({'a': i, 'b': i * 2}, ignore_index=True)\n"
                "result = out",
         modern="result = pd.DataFrame([{'a': i, 'b': i * 2} for i in range(3)], columns=['a', 'b'])"),
]

# Runs inside each env: executes snippets, records status and a version-neutral serialization.
CHILD = r'''
import json, math, sys, warnings
specs = json.load(sys.stdin)
DEPRECATIONS = (DeprecationWarning, FutureWarning, PendingDeprecationWarning)

def plain(v):
    import numpy as np
    if isinstance(v, (list, tuple)):
        return [plain(i) for i in v]
    if isinstance(v, dict):
        return {str(k): plain(i) for k, i in v.items()}
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float):
        if math.isnan(v): return "NaN"
        if math.isinf(v): return "inf" if v > 0 else "-inf"
        return v
    if v is None or isinstance(v, (bool, int, str)):
        return v
    return repr(v)

def ser(x):
    import numpy as np, pandas as pd
    if isinstance(x, pd.DataFrame):
        return {"t": "df", "columns": plain(list(x.columns)), "index": plain(list(x.index)),
                "index_names": plain(list(x.index.names)), "dtypes": [str(d) for d in x.dtypes],
                "values": plain(x.values.tolist())}
    if isinstance(x, pd.Series):
        return {"t": "series", "name": plain(x.name), "index": plain(list(x.index)),
                "index_names": plain(list(x.index.names)), "dtype": str(x.dtype),
                "values": plain(x.tolist())}
    if isinstance(x, pd.Index):
        return {"t": "index", "name": plain(x.name), "dtype": str(x.dtype), "values": plain(list(x))}
    if isinstance(x, np.ndarray):
        return {"t": "ndarray", "dtype": str(x.dtype), "shape": list(x.shape), "values": plain(x.tolist())}
    if isinstance(x, np.generic):
        return {"t": "npscalar", "dtype": str(x.dtype), "value": plain(x.item())}
    if isinstance(x, np.dtype):
        return {"t": "dtype", "value": str(x)}
    if isinstance(x, (list, tuple)):
        return {"t": "list", "items": [ser(i) for i in x]}
    if isinstance(x, dict):
        return {"t": "dict", "items": {str(k): ser(v) for k, v in x.items()}}
    return {"t": type(x).__name__, "value": plain(x)}

out = {}
for s in specs:
    ns = {}
    try:
        exec(s["setup"], ns)
    except Exception as e:
        out[s["key"]] = {"status": "FAIL", "detail": f"setup {type(e).__name__}: {e}"[:120]}
        continue
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            exec(s["code"], ns)
            dep = [w for w in caught if issubclass(w.category, DEPRECATIONS)]
            status = "WARN" if dep else "OK"
            detail = f"{dep[0].category.__name__}: {dep[0].message}"[:120] if dep else ""
            res = ser(ns["result"])
        except Exception as e:
            status, detail, res = "FAIL", f"{type(e).__name__}: {e}"[:120], None
    out[s["key"]] = {"status": status, "detail": detail, "result": res}
import numpy, pandas
out["__versions__"] = {"numpy": numpy.__version__, "pandas": pandas.__version__}
print(json.dumps(out))
'''


# One math thread per child: the snippets are tiny, and multithreaded BLAS in many parallel
# children exceeds per-user process limits (e.g., RLIMIT_NPROC on cluster login nodes).
SINGLE_THREAD = {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def run_env(python, specs):
    proc = subprocess.run([python, "-c", CHILD], input=json.dumps(specs),
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, **SINGLE_THREAD})
    if proc.returncode != 0:
        sys.exit(f"{python} failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def by_kind(dt):
    """'int64' -> 'i', 'float32' -> 'f', 'object' -> 'O' (rough numpy kind, no numpy needed)."""
    s = str(dt)
    for prefix, k in (("int", "i"), ("uint", "u"), ("float", "f"), ("bool", "b"),
                      ("complex", "c"), ("datetime", "M"), ("timedelta", "m")):
        if s.startswith(prefix):
            return k
    return s


def same(a, b, mode):
    """mode: 'exact' compares dtypes exactly, 'kind' by dtype kind, 'values' ignores dtypes."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        for k in a:
            if k in ("dtype", "dtypes") and mode != "exact":
                if mode == "values":
                    continue
                av = a[k] if isinstance(a[k], list) else [a[k]]
                bv = b[k] if isinstance(b[k], list) else [b[k]]
                if [by_kind(x) for x in av] != [by_kind(x) for x in bv]:
                    return False
            elif not same(a[k], b[k], mode):
                return False
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, mode) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    return a == b


def match_level(a, b):
    if a is None or b is None:
        return "—"
    for mode, label in (("exact", "exact"), ("kind", "dtype width differs"),
                        ("values", "DTYPE KIND DIFFERS")):
        if same(a, b, mode):
            return label
    return "VALUES DIFFER"


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=str(root / "envs"))
    ap.add_argument("--out", default=str(root / "tasks"))
    args = ap.parse_args()
    envs = Path(args.envs)

    legacy_specs = [dict(key=p["id"], setup=p["setup"], code=p["legacy"]) for p in PROBES]
    modern_specs = legacy_specs + [dict(key=p["id"] + "::rewrite", setup=p["setup"], code=p["modern"])
                                   for p in PROBES]
    L = run_env(str(envs / "legacy" / "bin" / "python"), legacy_specs)
    M = run_env(str(envs / "modern" / "bin" / "python"), modern_specs)

    rows = []
    for p in PROBES:
        lg, mo, rw = L[p["id"]], M[p["id"]], M[p["id"] + "::rewrite"]
        if lg["status"] == "FAIL":
            verdict = "BROKEN PROBE (legacy fails)"
        elif rw["status"] != "OK":
            verdict = "REWRITE NOT CLEAN"
        elif mo["status"] in ("FAIL", "WARN"):
            verdict = "detectable"
        elif match_level(lg["result"], mo["result"]) != "exact":
            verdict = "SILENT change"
        else:
            verdict = "no migration needed"
        rows.append(dict(id=p["id"], kind=p["kind"], lib=p["lib"], history=p["history"],
                         legacy=lg["status"], modern=mo["status"], rewrite=rw["status"],
                         equivalence=match_level(lg.get("result"), rw.get("result")),
                         verdict=verdict,
                         legacy_detail=lg["detail"], modern_detail=mo["detail"],
                         rewrite_detail=rw["detail"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"legacy": L["__versions__"], "modern": M["__versions__"]}
    (out / "migration_types.json").write_text(json.dumps({"versions": meta, "probes": rows}, indent=2))

    lines = [f"# Migration types — verified against legacy "
             f"(pandas {meta['legacy']['pandas']}, NumPy {meta['legacy']['numpy']}) and modern "
             f"(pandas {meta['modern']['pandas']}, NumPy {meta['modern']['numpy']})", "",
             "Generated by tasks/verify_types.py. History notes verified 2026-10-22 against official release notes.", "",
             "| id | kind | legacy | modern (orig) | rewrite | rewrite vs legacy | verdict | history |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['kind']} | {r['legacy']} | {r['modern']} | {r['rewrite']} | "
                     f"{r['equivalence']} | {r['verdict']} | {r['history']} |")
    lines += ["", "## Details (warnings and errors)", ""]
    for r in rows:
        for side in ("legacy", "modern", "rewrite"):
            d = r[f"{side}_detail"]
            if d:
                lines.append(f"- **{r['id']}** ({side}): {d.replace('|', '/')}")
    (out / "migration_types.md").write_text("\n".join(lines) + "\n")

    w = max(len(r["id"]) for r in rows)
    print(f"{'id':{w}}  {'kind':8}  legacy  modern  rewrite  {'rewrite vs legacy':20}  verdict")
    for r in rows:
        print(f"{r['id']:{w}}  {r['kind']:8}  {r['legacy']:6}  {r['modern']:6}  {r['rewrite']:7}  "
              f"{r['equivalence']:20}  {r['verdict']}")
    print(f"\nWrote {out / 'migration_types.md'} and migration_types.json")


if __name__ == "__main__":
    main()
