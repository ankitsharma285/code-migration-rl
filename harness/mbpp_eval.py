"""MBPP regression check: did migration training hurt general Python coding?

    python -m harness.mbpp_eval --base-url http://localhost:8000/v1 --model policy --out results/final/mbpp_base
    python -m harness.mbpp_eval --compare results/final/mbpp_tp_s1,results/final/mbpp_tp_s2 results/final/mbpp_base

Data: the sanitized MBPP release (Austin et al. 2021), test problems 11-510, fetched once to data/mbpp/:
    mkdir -p data/mbpp && curl -L -o data/mbpp/sanitized-mbpp.json \\
      https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json
Prompt: the problem text plus its assert tests (the standard MBPP setup), answer in one ```python block.
Greedy decoding, one sample, scored by running the asserts in the MODERN sandbox runtime (no AST check: MBPP
code legitimately uses any module). A problem passes if every assert holds.
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from env.reward import extract_code
from env.runner import Limits, run_job
from harness import manifest
from harness.policies import OpenAIPolicy

DATA = Path(__file__).resolve().parent.parent / "data" / "mbpp" / "sanitized-mbpp.json"
SYSTEM = "You are an expert Python programmer."


def problems():
    if not DATA.exists():
        sys.exit(f"missing {DATA}; see the download command in this file's docstring")
    rows = json.load(open(DATA))
    return [r for r in rows if 11 <= r["task_id"] <= 510]


def prompt(p):
    tests = "\n".join(p["test_list"])
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{p['prompt']}\nYour code should pass these tests:\n\n{tests}\n\n"
                                        "Reply with the function in a single ```python code block."}]


def program(p, code):
    body = "\n".join("    " + t for t in p["test_list"])
    return "\n".join(p.get("test_imports") or []) + "\n" + code + f"\n\ndef run_mbpp_tests():\n{body}\n    return True\n"


def run(args):
    ps = problems()[: args.limit] if args.limit else problems()
    record = manifest.build({k: v for k, v in vars(args).items() if k != "out"}, ps)
    if record["git"]["dirty"] and not args.allow_dirty:
        sys.exit("Refusing to run: uncommitted changes.\n" + "\n".join(record["git"]["dirty_files"]))
    policy = OpenAIPolicy(args.model, args.base_url, temperature=0.0, max_tokens=args.max_tokens)
    with ThreadPoolExecutor(args.gen_workers) as pool:
        replies = list(pool.map(lambda p: policy._one(prompt(p)), ps))
    limits = Limits(exec_s=20, cpu_s=20)   # a few reference solutions need >10 s

    def score(p, reply):
        code = extract_code(reply)
        if code is None:
            return "no_code", ""
        res = run_job("modern", program(p, code), "run_mbpp_tests", [{}], limits)
        if res.status != "ok":
            return res.status, res.detail
        if res.define.get("status") == "FAIL":
            return "define_error", res.define.get("detail", "")
        c = res.calls[0]
        ok = c["status"] in ("OK", "WARN") and c.get("result", {}).get("value") is True
        return ("pass" if ok else "fail"), c.get("detail", "")

    with ThreadPoolExecutor(args.workers) as pool:
        scored = list(pool.map(lambda pr: score(*pr), zip(ps, replies)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for p, reply, (reason, detail) in zip(ps, replies, scored):
            f.write(json.dumps({"task_id": p["task_id"], "pass": reason == "pass", "reason": reason,
                                "detail": detail[:300], "reply": reply}) + "\n")
    n_pass = sum(r == "pass" for r, _ in scored)
    reasons = {}
    for r, _ in scored:
        reasons[r] = reasons.get(r, 0) + 1
    record["result"] = {"pass_at_1": n_pass / len(ps), "n": len(ps), "reasons": reasons}
    (out / "run.json").write_text(json.dumps(record, indent=1))
    print(f"MBPP pass@1 (greedy) {n_pass / len(ps):.3f} on {len(ps)} problems; reasons {reasons}; "
          f"commit {(record['git']['commit'] or 'none')[:8]} -> {out}")


def per_task(dirs):
    runs = [{json.loads(l)["task_id"]: json.loads(l)["pass"] for l in open(Path(d) / "results.jsonl")}
            for d in dirs.split(",")]
    common = set.intersection(*(set(r) for r in runs))
    return {t: sum(r[t] for r in runs) / len(runs) for t in common}


def compare(a, b):
    pa, pb = per_task(a), per_task(b)
    common = sorted(set(pa) & set(pb))
    diffs = [pa[t] - pb[t] for t in common]
    rng = random.Random(0)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
    ma, mb = sum(pa[t] for t in common) / len(common), sum(pb[t] for t in common) / len(common)
    print(f"A = {a}\nB = {b}\n{len(common)} problems\nA {ma:.1%}   B {mb:.1%}   A-B {ma - mb:+.1%}   "
          f"95% CI [{means[250]:+.1%}, {means[9750]:+.1%}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="policy")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--gen-workers", type=int, default=32)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
