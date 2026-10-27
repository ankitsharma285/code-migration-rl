"""Stage 7: re-score stored completions on the EVALUATION hidden seeds.

    python -m harness.rescore_runs results/s1_ck00150_mt3_heldout_id results/tp_ck00150_mt3_heldout_id ...
        [--seeds eval:final] [--out-dir results/final] [--workers 30]

Every number so far used the training-time hidden inputs (TRAIN_HIDDEN_SEEDS), the same inputs the reward was
computed on during training and the ones every checkpoint and config choice was made on. The evaluation seeds
(env/inputs.eval_seeds) were never used for anything. Re-scoring needs no GPU: the stored replies are scored
again on new hidden inputs. Visible inputs are unchanged, so every feedback message and every stop decision in a
multi-turn episode is exactly what the model saw; only the hidden scoring is new.

Writes <out-dir>/<run name>/ (same layout as any eval run). Runs that already have an output are skipped.
"""
import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

print = functools.partial(print, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--seeds", default="eval:final")
    ap.add_argument("--out-dir", default="results/final")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    failed = []
    for run in args.runs:
        run = Path(run.rstrip("/"))
        out = Path(args.out_dir) / run.name
        if (out / "metrics.json").exists():
            print(f"skip {run} (already re-scored at {out})")
            continue
        cfg = json.load(open(run / "run.json"))["config"]
        cmd = [sys.executable, "-m", "harness.evaluate", "--rescore", str(run / "trajectories.jsonl"),
               "--tasks", cfg["tasks"], "--seeds", args.seeds, "--workers", str(args.workers), "--out", str(out)]
        for flag in ("turns",):                     # multi-turn re-scoring needs the episode budget
            if cfg.get(flag):
                cmd += [f"--{flag}", str(cfg[flag])]
        print(f"\n== {run} ({cfg['tasks']}) -> {out}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(str(run))
    if failed:
        sys.exit(f"\nFAILED: {failed}")


if __name__ == "__main__":
    main()
