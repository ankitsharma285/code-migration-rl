"""Stage 7 "unfiltered" held-out sets: tasks the filter dropped ONLY because the generator's own migration failed.

    envs/legacy/bin/python -m tasks.build_hard_split --raw tasks/generated/*/raw*.jsonl \\
        --rejected tasks/generated/*/rejected.jsonl [--workers 30]

The last filter step (tasks/filter_tasks.py) keeps a task only if the generating LLM's reference migration scores
1. That makes every kept task solvable, but it may also make the benchmark easier: tasks the generator itself got
wrong are dropped. Scoring never uses the reference migration (it compares against the LEGACY output), so those
tasks are still valid, as long as they pass every other check, which this script re-runs. Solvability is not
verified, so read results on these sets relatively (model vs model), not as absolute ceilings.

Writes tasks/splits/hard_id.jsonl (training migration types) and hard_cat.jsonl (held-out types), deduplicated
against every frozen split and against each other, plus tasks/splits/hard_manifest.json.
"""
import argparse
import glob
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tasks.build_splits import jaccard, shingles
from tasks.filter_tasks import HELD_OUT, Reject, normalized_hash, verify

SPLITS = Path(__file__).resolve().parent / "splits"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--rejected", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    expand = lambda pats: sorted({p for pat in pats for p in glob.glob(pat)})

    want = set()
    for path in expand(args.rejected):
        for line in open(path):
            r = json.loads(line)
            if (r.get("reason") or "").startswith("migration:"):
                want.add(r["request_id"])
    recs = [json.loads(l) for path in expand(args.raw) for l in open(path)]
    recs = [r for r in recs if r.get("request_id") in want]
    print(f"{len(want)} tasks rejected only at the migration check; {len(recs)} found in the raw files")

    def one(rec):
        try:
            return verify(rec, rec.get("split", "train"), require_migration=False)
        except Reject as e:
            return str(e)
    with ThreadPoolExecutor(args.workers) as pool:
        out = list(pool.map(one, recs))
    kept = [t for t in out if isinstance(t, dict) and not t["reference_ok"]]
    print(f"re-verified: {len(kept)} pass every other check (dropped {len(out) - len(kept)})")

    frozen = [json.loads(l) for n in ("train", "heldout_id", "heldout_cat") for l in open(SPLITS / f"{n}.jsonl")]
    hashes = {normalized_hash(t["legacy"]) for t in frozen}
    sh_frozen = [shingles(t["legacy"]) for t in frozen]
    final, sh_kept = [], []
    for t in kept:
        s = shingles(t["legacy"])
        h = normalized_hash(t["legacy"])
        if h in hashes or any(jaccard(s, x) >= args.threshold for x in sh_frozen + sh_kept):
            continue
        hashes.add(h)
        sh_kept.append(s)
        final.append(t)
    hard_id = [t for t in final if not set(t["types"]) & HELD_OUT]
    hard_cat = [t for t in final if set(t["types"]) & HELD_OUT]
    manifest = {"source": "filter rejections at the migration check only", "threshold": args.threshold}
    for name, rows in (("hard_id", hard_id), ("hard_cat", hard_cat)):
        with open(SPLITS / f"{name}.jsonl", "w") as f:
            for t in rows:
                f.write(json.dumps(t) + "\n")
        h = hashlib.sha256()
        for t in rows:
            h.update(json.dumps(t, sort_keys=True).encode())
        manifest[name] = {"n": len(rows), "sha256": h.hexdigest(),
                          "reference_failure": dict(sorted({r: sum(t["reference_reason"] == r for t in rows)
                                                            for r in {t["reference_reason"] for t in rows}}.items()))}
    (SPLITS / "hard_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
