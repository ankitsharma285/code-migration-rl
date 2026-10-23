"""Print a few accepted tasks for manual review: the filter guarantees validity, not quality.

Usage: envs/legacy/bin/python -m tasks.show_tasks tasks/generated/<run>/accepted.jsonl [--n 5] [--seed 0]
"""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
tasks = [json.loads(l) for l in open(a.path)]
for t in random.Random(a.seed).sample(tasks, min(a.n, len(tasks))):
    print("=" * 80)
    print(f"{t['id']}  domain={t['domain']}  types={t['types']}  coverage={t['coverage']}")
    print("-" * 36 + " LEGACY " + "-" * 36)
    print(t["legacy"].rstrip())
    print("-" * 36 + " MODERN " + "-" * 36)
    print(t["modern"].rstrip())
