"""Stage 2 exit check.

1. Every seed task's hand-written migration scores 1.0.
2. Every seed task's unchanged legacy code scores 0.0 (with its reason code).
3. A missing code block gets the malformed penalty.
4. Sandbox abuse cases are contained and scored 0 (or 1 where the code is legitimately correct).

Usage (from the repo root):  envs/legacy/bin/python -m env.check_stage2
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from env.inputs import TRAIN_HIDDEN_SEEDS, make_inputs
from env.reward import MALFORMED_PENALTY, compute_references, score
from env.runner import network_mode
from tasks.seed_tasks import SEED_TASKS


def fence(code):
    return f"Here is the migrated code:\n\n```python\n{code.strip()}\n```\n"


S01 = next(t for t in SEED_TASKS if t["id"] == "s01_group_summary")
S03 = next(t for t in SEED_TASKS if t["id"] == "s03_scale_cells")

# (name, task, completion, expected_reward, expected_reason)
ABUSE = [
    ("infinite loop", S01, fence("def summarize(df):\n    while True:\n        pass"), 0.0, "timeout"),
    ("memory bomb", S01, fence("def summarize(df):\n    x = bytearray(10**11)\n    return x"), 0.0, "crash"),
    ("network access", S01, fence("import socket\ndef summarize(df):\n"
                                  "    socket.create_connection(('pypi.org', 443), timeout=3)\n"), 0.0, "crash"),
    ("calls exit()", S01, fence("def summarize(df):\n    exit(0)"), 0.0, "crash"),
    ("hard exit os._exit", S01, fence("import os\ndef summarize(df):\n    os._exit(0)"), 0.0, "crash"),
    ("forged result + os._exit", S03, fence(
        "import json, os, sys\n"
        "def scale_cells(df):\n"
        "    r = df.applymap(lambda x: float(x) / 100)   # deprecated: warns in modern\n"
        "    sys.__stdout__.write(json.dumps({'define': {'status': 'OK', 'detail': '', 'warning': ''},\n"
        "        'calls': [{'status': 'OK', 'detail': '', 'warning': '', 'result': None}]}))\n"
        "    sys.__stdout__.flush()\n"
        "    os._exit(0)"), 0.0, "crash"),
    ("prints a lot, correct", S01, fence("def summarize(df):\n    for i in range(100000):\n        print(i)\n"
                                         + S01["modern"].replace("def summarize(df):", "")), 1.0, "pass"),
    ("wrong function name", S01, fence(S01["modern"].replace("def summarize", "def summary")), 0.0, "missing_entry"),
    ("syntax error", S01, fence("def summarize(df)\n    return df"), 0.0, "syntax"),
    ("no code block", S01, "I would use pd.concat instead of append.", MALFORMED_PENALTY, "no_code"),
]


def main():
    print(f"network isolation: {network_mode()}")
    t0 = time.monotonic()
    inputs = {t["id"]: [make_inputs(t, s) for s in TRAIN_HIDDEN_SEEDS] for t in SEED_TASKS}
    with ThreadPoolExecutor(8) as pool:
        refs = dict(zip(inputs, pool.map(lambda t: compute_references(t, inputs[t["id"]]), SEED_TASKS)))
    print(f"references for {len(SEED_TASKS)} tasks x {len(TRAIN_HIDDEN_SEEDS)} inputs: {time.monotonic() - t0:.1f}s")

    cases = []
    for t in SEED_TASKS:
        cases.append((f"{t['id']} correct", t, fence(t["modern"]), 1.0, None))
        cases.append((f"{t['id']} unchanged legacy", t, fence(t["legacy"]), 0.0, None))
    cases += ABUSE

    t0 = time.monotonic()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda c: score(c[1], c[2], inputs[c[1]["id"]], refs[c[1]["id"]]), cases))
    elapsed = time.monotonic() - t0

    failures = 0
    print(f"\n{'case':42s} {'reward':>6s}  reason         detail")
    for (name, _, _, want_r, want_reason), r in zip(cases, results):
        ok = r.reward == want_r and (want_reason is None or r.reason == want_reason)
        failures += not ok
        flag = "" if ok else "   <-- UNEXPECTED"
        print(f"{name:42s} {r.reward:6.1f}  {r.reason:13s}  {r.detail[:60]}{flag}")

    walls = sorted(r.wall_s for r in results if r.wall_s)
    starts = sorted(r.startup_s for r in results if r.startup_s)
    print(f"\n{len(cases)} cases scored in {elapsed:.1f}s with 8 workers; "
          f"per-job wall time median {walls[len(walls) // 2]:.2f}s, max {walls[-1]:.2f}s")
    if starts:
        print(f"import time (pandas + NumPy) per job: median {starts[len(starts) // 2]:.2f}s, "
              f"max {starts[-1]:.2f}s  <- filesystem cost, paid by every candidate")
    print("PASS" if not failures else f"FAIL: {failures} unexpected results")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
