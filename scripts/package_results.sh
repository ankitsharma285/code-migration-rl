#!/bin/bash
# Bundle what a reader needs to re-derive every reported number WITHOUT a GPU: the evaluation runs
# (stored replies + records) and the training logs. No checkpoints (~3 GB each).
#   bash scripts/package_results.sh            -> results_bundle.tar.gz (attach to a GitHub release)
set -euo pipefail
OUT=${OUT:-results_bundle.tar.gz}
files=$(
  find results/final -maxdepth 2 -type f \( -name run.json -o -name metrics.json -o -name timing.json \
       -o -name trajectories.jsonl -o -name completions.jsonl -o -name '*.md' -o -name '*.json' -o -name '*.txt' \)
  for r in mt_s1 mt_s2 mt_tp_s1 mt_tp_s2 st_s1 st_s2 rft_s1 rft_s2; do
    for f in config.json run.json log.jsonl rollouts.jsonl; do [ -f "results/$r/$f" ] && echo "results/$r/$f"; done
  done
  [ -f results/profile_tp/profile/top_ops.txt ] && echo results/profile_tp/profile/top_ops.txt results/profile_tp/profile/profile.json
  for f in results/rft/sft_data.jsonl results/rft/sft_data.stats.json results/rft2/sft_data.jsonl results/rft2/sft_data.stats.json; do
    [ -f "$f" ] && echo "$f"; done
  true    # the substitution must not inherit a failed [ -f ] from the last iteration (set -e)
)
# Scrub machine-specific paths and the account name from a COPY of every file. They appear in warning and
# traceback text (pandas reports its own file path) and in run records; no score, metric or replayed
# input depends on them, so re-scoring the bundle gives the same numbers.
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/bundle_XXXX")
trap 'rm -rf "$STAGE"' EXIT
envs/legacy/bin/python - "$STAGE" "$PWD" "$HOME" "$USER" $files <<'PY'
import os, sys
stage, repo, home, user, *files = sys.argv[1:]
subs = [(os.path.realpath(repo), "<repo>"), (repo, "<repo>"), (os.path.realpath(home), "<home>"),
        (home, "<home>"), (user, "<user>")]
subs = sorted({(a, b) for a, b in subs if a}, key=lambda x: -len(x[0]))
for f in files:
    text = open(f, encoding="utf-8", errors="surrogateescape").read()
    for a, b in subs:
        text = text.replace(a, b)
    out = os.path.join(stage, f)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8", errors="surrogateescape").write(text)
PY
tar czf "$OUT" --owner=0 --group=0 --numeric-owner -C "$STAGE" $files
echo "$OUT: $(du -h "$OUT" | cut -f1), $(echo "$files" | wc -w) files"
left=$(zcat "$OUT" | grep -a -c -E "$USER|lustre" || true)
echo "lines still mentioning \$USER or lustre: $left"
