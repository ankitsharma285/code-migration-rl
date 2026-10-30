#!/bin/bash
# Run before making the repo public. Prints anything that should not be published; silence = clean.
set -uo pipefail
PAT='lustre|an669323|/home/[a-z]|evuser|evc[0-9]|api[_-]?key *[=:] *["'"'"'][^"'"'"'$]|sk-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{20}|BEGIN (RSA|OPENSSH) PRIVATE'
echo "== tracked files matching cluster paths / secrets"
git grep -n -I -E "$PAT" -- . ':!scripts/prepublish_check.sh' || true
echo "== the same patterns anywhere in git history (old versions of files)"
git log -p --all -- . ':!scripts/prepublish_check.sh' | grep -n -E "^\+.*($PAT)" | head -20 || true
echo "== tracked files over 5 MB"
git ls-files -z | xargs -0 du -k 2>/dev/null | awk '$1 > 5120 {print $1 " KB  " $2}'
echo "== required files"
for f in README.md LICENSE envs/legacy.lock envs/modern.lock tasks/splits/manifest.json tasks/splits/train.jsonl \
         tasks/splits/heldout_id.jsonl tasks/splits/heldout_cat.jsonl tasks/splits/hard_id.jsonl tasks/splits/hard_cat.jsonl; do
  git ls-files --error-unmatch "$f" > /dev/null 2>&1 || echo "NOT TRACKED: $f"
done
echo "== uncommitted changes"
git status --short --untracked-files=no
