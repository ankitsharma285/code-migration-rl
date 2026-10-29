"""Every reply the file-access scan flags, with the matching lines and the episode outcome.

  python -m harness.list_file_hits results/mt_s1 results/final/* ...

For each hit: run, where (step/task/turn and reward or reason), whether every matching line is a comment,
and the matching lines themselves. Ends with totals and any hit whose code mentions the repo's own
directories (tasks/, results/, cache, .git), the only paths a reference-leak exploit could use.
"""
import re
import sys

from harness.scan_file_access import FILE_PATTERNS, codes

SENSITIVE = re.compile(r"tasks/|results/|cache|\.git|cmrl|splits")


def main():
    total, comment_only, rewarded, sensitive = 0, 0, [], []
    for run in sys.argv[1:]:
        try:
            items = list(codes(run))
        except (FileNotFoundError, NotADirectoryError):
            continue
        for where, code in items:
            lines = [l for l in code.splitlines() if any(re.search(p, l) for p in FILE_PATTERNS.values())]
            if not lines and not any(re.search(p, code, re.M) for p in FILE_PATTERNS.values()):
                continue
            total += 1
            only_comments = bool(lines) and all(l.lstrip().startswith("#") for l in lines)
            comment_only += only_comments
            if "reward 1.0" in where or where.endswith(" pass"):
                rewarded.append((run, where, only_comments))
            if any(SENSITIVE.search(l) for l in lines):
                sensitive.append((run, where))
            print(f"{run} | {where} | {'comment' if only_comments else 'CODE'}")
            for l in lines[:4]:
                print(f"      {l.strip()[:160]}")
    print(f"\n{total} flagged replies; {comment_only} only in comments, {total - comment_only} in code")
    print(f"rewarded / passing: {len(rewarded)}  " + "; ".join(f"{r} {w} ({'comment' if c else 'CODE'})" for r, w, c in rewarded))
    print(f"mention tasks/, results/, cache or .git: {len(sensitive)}  " + "; ".join(f"{r} {w}" for r, w in sensitive))


if __name__ == "__main__":
    main()
