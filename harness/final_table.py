"""Stage 7 results table: every arm on every split, pooled over seeds, with 95% bootstrap CIs over tasks.

    python -m harness.final_table configs/final_table.json [--dir results/final] [--out results/final/table]

The spec lists rows (an arm: label, run names per split, best_of K for single-turn arms) and comparisons
(row A minus row B on a split). Run names are folders under --dir; "a,b" pools seeds (per-task mean over runs).
A row whose runs are missing prints "-" (so the table can be built while jobs are still running).

Columns per split: final pass, turn-1 pass, and final pass on rename vs behavior-change tasks (the task's
"behavior" flag from tasks/filter_tasks.py). Writes <out>.md and <out>.json.
"""
import argparse
import json
from pathlib import Path

from harness.compare_runs import bootstrap, load_many
from harness.evaluate import load_tasks


def cell(dirs, split_dir, best_of=1):
    paths = [str(split_dir / d) for d in dirs.split(",")]
    if not all(Path(p, "trajectories.jsonl").exists() for p in paths):
        return None
    return load_many(",".join(paths), best_of)[1]


def mean_ci(per_task, ids, key):
    ids = [t for t in ids if t in per_task]
    if not ids:
        return None
    vals = [per_task[t][key] for t in ids]
    lo, hi = bootstrap(vals)
    return sum(vals) / len(vals), lo, hi, len(ids)


def fmt(x, ci=True):
    if x is None:
        return "-"
    m, lo, hi, _ = x
    return f"{m:.1%} [{lo:.1%}, {hi:.1%}]" if ci else f"{m:.1%}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--dir", default="results/final")
    ap.add_argument("--out", default="results/final/table")
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    base = Path(args.dir)
    out_json, lines = {"rows": {}, "comparisons": []}, []

    for split in spec["splits"]:
        try:
            tasks = load_tasks(split)
        except SystemExit:
            lines.append(f"\n## {split}: no task file yet (tasks/splits/{split}.jsonl)\n")
            continue
        ids = [t["id"] for t in tasks]
        behavior = {t["id"] for t in tasks if t.get("behavior")}
        lines += [f"\n## {split} ({len(ids)} tasks)\n",
                  "| Arm | Final pass [95% CI] | Turn 1 | Rename tasks | Behavior-change tasks |",
                  "|---|---|---|---|---|"]
        for row in spec["rows"]:
            runs = row["runs"].get(split)
            pt = cell(runs, base, row.get("best_of", 1)) if runs else None
            if pt is None:
                lines.append(f"| {row['label']} | - | - | - | - |")
                continue
            final = mean_ci(pt, ids, "final")
            turn1 = mean_ci(pt, ids, "turn1") if row.get("best_of", 1) == 1 else None
            ren = mean_ci(pt, [t for t in ids if t not in behavior], "final")
            beh = mean_ci(pt, [t for t in ids if t in behavior], "final")
            out_json["rows"].setdefault(row["label"], {})[split] = {"final": final, "turn1": turn1,
                                                                    "rename": ren, "behavior": beh}
            lines.append(f"| {row['label']} | {fmt(final)} | {fmt(turn1, False)} | {fmt(ren, False)} "
                         f"(n={ren[3] if ren else 0}) | {fmt(beh, False)} (n={beh[3] if beh else 0}) |")

    lines += ["\n## Comparisons (A minus B, paired over tasks, seeds pooled)\n",
              "| Split | A | B | A - B [95% CI] |", "|---|---|---|---|"]
    rows = {r["label"]: r for r in spec["rows"]}
    for c in spec["comparisons"]:
        a, b, split = rows[c["a"]], rows[c["b"]], c["split"]
        pa = cell(a["runs"].get(split, ""), base, a.get("best_of", 1)) if a["runs"].get(split) else None
        pb = cell(b["runs"].get(split, ""), base, b.get("best_of", 1)) if b["runs"].get(split) else None
        if pa is None or pb is None:
            lines.append(f"| {split} | {c['a']} | {c['b']} | - |")
            continue
        common = sorted(set(pa) & set(pb))
        diffs = [pa[t]["final"] - pb[t]["final"] for t in common]
        d = sum(diffs) / len(diffs)
        lo, hi = bootstrap(diffs)
        out_json["comparisons"].append({**c, "diff": d, "ci": [lo, hi], "n": len(common)})
        lines.append(f"| {split} | {c['a']} | {c['b']} | {d:+.1%} [{lo:+.1%}, {hi:+.1%}] |")

    text = "# Final results (evaluation seeds)\n" + "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.md").write_text(text)
    Path(f"{args.out}.json").write_text(json.dumps(out_json, indent=1))
    print(text)


if __name__ == "__main__":
    main()
