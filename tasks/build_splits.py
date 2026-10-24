"""Freeze the task splits from filtered tasks.

  1. Pool every accepted task (train-type runs and held-out-type runs).
  2. Near-duplicate clustering: tokens are normalized (local names and literals masked; pandas/NumPy
     API names kept), shingled, and compared by Jaccard similarity; pairs above --threshold join one
     cluster (union-find). Exact structural duplicates were already removed by the filter.
  3. Drop tasks that near-copy a seed task (seed tasks were shown to the generator as format
     examples and serve as few-shot examples for the baseline).
  4. Split train-type tasks BY CLUSTER, so near-twins never straddle train and test:
       heldout_id   ~--heldout-id tasks, the main metric (in-distribution, unseen tasks)
       train        everything else
     Held-out-type tasks (mad / weekofyear / lookup) form heldout_cat, the generalization test.
  5. Write tasks/splits/{train,heldout_id,heldout_cat}.jsonl and manifest.json with sha256 hashes.

Usage (from the repo root):
  envs/legacy/bin/python -m tasks.build_splits tasks/generated/all_train/accepted.jsonl \
      tasks/generated/all_heldout/accepted.jsonl --out tasks/splits
"""
import argparse
import hashlib
import io
import json
import keyword
import random
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

from tasks.seed_tasks import SEED_TASKS

HERE = Path(__file__).resolve().parent
CATS = json.loads((HERE / "categories.json").read_text())
HELD_OUT = set(CATS["held_out"])
KEEP_NAMES = {"pd", "np", "io", "self"}


def tokens(code):
    """Normalized tokens: keywords, operators, and attribute/API names kept; other names -> N, literals -> L."""
    out, prev = [], None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.NAME:
                keep = keyword.iskeyword(tok.string) or tok.string in KEEP_NAMES or prev == "."
                out.append(tok.string if keep else "N")
            elif tok.type in (tokenize.NUMBER, tokenize.STRING):
                out.append("L")
            elif tok.type == tokenize.OP:
                out.append(tok.string)
            prev = tok.string if tok.type in (tokenize.OP, tokenize.NAME) else prev
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def shingles(code, k=5):
    t = tokens(code)
    return {tuple(t[i:i + k]) for i in range(max(1, len(t) - k + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def cluster(tasks, threshold):
    sh = [shingles(t["legacy"]) for t in tasks]
    # candidate pairs share a rare shingle (inverted index) -> avoids comparing every pair
    index = defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            index[g].append(i)
    uf, checked = UnionFind(len(tasks)), set()
    for members in index.values():
        if len(members) > 200:                     # very common shingle: uninformative
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if (i, j) in checked:
                    continue
                checked.add((i, j))
                if jaccard(sh[i], sh[j]) >= threshold:
                    uf.union(i, j)
    groups = defaultdict(list)
    for i in range(len(tasks)):
        groups[uf.find(i)].append(i)
    return list(groups.values()), sh


def sha256_lines(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accepted", nargs="+", help="accepted.jsonl files from tasks.filter_tasks")
    ap.add_argument("--out", default=str(HERE / "splits"))
    ap.add_argument("--heldout-id", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.7, help="Jaccard similarity for near-duplicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-test-cluster", type=int, default=5,
                    help="clusters larger than this stay in train (one pattern must not dominate the test split)")
    args = ap.parse_args()

    pool, seen_ids = [], set()
    for path in args.accepted:
        for line in open(path):
            t = json.loads(line)
            if t["id"] not in seen_ids:
                seen_ids.add(t["id"])
                pool.append(t)

    # seed near-copies
    seed_sh = [shingles(s["legacy"]) for s in SEED_TASKS]
    def near_seed(t):
        s = shingles(t["legacy"])
        return max(jaccard(s, x) for x in seed_sh) >= args.threshold
    seed_copies = [t["id"] for t in pool if near_seed(t)]
    pool = [t for t in pool if t["id"] not in set(seed_copies)]

    cat = [t for t in pool if set(t["types"]) & HELD_OUT]
    trainable = [t for t in pool if not set(t["types"]) & HELD_OUT]

    # cluster each group separately; a held-out-type task can never enter train anyway
    clusters, _ = cluster(trainable, args.threshold)
    # Random cluster order, so the test split has the same mix of tasks as train (filling from small
    # clusters first would make it all unusual one-offs). Only very large clusters -- one pattern
    # repeated many times -- are kept out of the test split, so no single pattern dominates it.
    rng = random.Random(args.seed)
    rng.shuffle(clusters)
    test_idx, filled = set(), 0
    for c in clusters:
        if filled >= args.heldout_id:
            break
        if len(c) > args.max_test_cluster:
            continue
        test_idx.update(c)
        filled += len(c)
    heldout_id = [trainable[i] for i in sorted(test_idx)]
    train = [trainable[i] for i in range(len(trainable)) if i not in test_idx]

    cat_clusters, _ = cluster(cat, args.threshold) if cat else ([], None)
    # one task per near-duplicate cluster in heldout_cat: repeated near-twins would overweight a pattern
    heldout_cat = [cat[min(c)] for c in cat_clusters]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "heldout_id": heldout_id, "heldout_cat": heldout_cat}
    for name, rows in splits.items():
        rows.sort(key=lambda t: t["id"])
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    sizes = Counter(len(c) for c in clusters)
    per_type = {name: dict(sorted(Counter(ty for t in rows for ty in t["types"]).items()))
                for name, rows in splits.items()}
    missing_in_test = sorted(set(per_type["train"]) - set(per_type["heldout_id"]))
    manifest = {
        "built_from": args.accepted, "threshold": args.threshold, "seed": args.seed,
        "pool": len(pool) + len(seed_copies), "dropped_seed_copies": len(seed_copies),
        "near_duplicate_clusters": len(clusters),
        "tasks_in_multi_task_clusters": sum(len(c) for c in clusters if len(c) > 1),
        "largest_cluster": max(sizes) if sizes else 0,
        "heldout_cat_dropped_near_twins": len(cat) - len(heldout_cat),
    "cluster_sizes": dict(sorted(sizes.items())),
        "sizes": {k: len(v) for k, v in splits.items()},
        "sha256": {k: sha256_lines(v) for k, v in splits.items()},
        "behavior_share": {k: round(sum(t.get("behavior", False) for t in v) / max(len(v), 1), 3)
                           for k, v in splits.items()},
        "types_missing_from_heldout_id": missing_in_test,
        "per_type": per_type,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_type"}, indent=1))


if __name__ == "__main__":
    main()
