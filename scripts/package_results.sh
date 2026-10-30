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
)
tar czf "$OUT" --owner=0 --group=0 --numeric-owner $files
echo "$OUT: $(du -h "$OUT" | cut -f1), $(echo "$files" | wc -w) files"
