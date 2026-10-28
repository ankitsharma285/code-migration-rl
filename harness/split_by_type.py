"""Pass rate on tasks with vs without one migration type, for several arms.

  python -m harness.split_by_type --split hard_id --type level_reduction \
      codemod=codemod_hard_id tp=tp_ck00150_mt3_hard_id,tp2_ck00150_mt3_hard_id ...

Each arm is label=run[,run...] under results/final (seeds pooled as per-task means).
"""
import argparse
from harness.compare_runs import load_many
from harness.evaluate import load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--root", default="results/final")
    ap.add_argument("--best-of", type=int, default=1)
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    types = {t["id"]: t["types"] for t in load_tasks(a.split)}
    has = {t for t, ty in types.items() if a.type in ty}
    print(f"{a.split}: {len(has)} tasks with {a.type}, {len(types) - len(has)} without\n")
    print(f"| Arm | with {a.type} | without | all |\n|---|---|---|---|")
    for arm in a.arms:
        label, runs = arm.split("=", 1)
        _, r = load_many(",".join(f"{a.root}/{x}" for x in runs.split(",")), a.best_of)
        rate = lambda ts: f"{100 * sum(r[t]['final'] for t in ts) / len(ts):.1f}% (n={len(ts)})" if ts else "-"
        w = [t for t in r if t in has]; wo = [t for t in r if t not in has]
        print(f"| {label} | {rate(w)} | {rate(wo)} | {rate(list(r))} |")


if __name__ == "__main__":
    main()
