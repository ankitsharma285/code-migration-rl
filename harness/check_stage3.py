"""Stage 3 gate, part 1 (part 2 is reproducing a result: harness/reproduce.py).

  1. AST false positives: every correct seed migration must pass the AST check.
  2. Cheater suite: every cheater must score 0 on every seed task. Also re-scored with the AST check
     OFF, to show which defense stops each cheater (informational: these are the holes it closes).
  3. Leakage audit: no hidden input or hidden reference output appears in any prompt.

Usage (from the repo root):  envs/legacy/bin/python -m harness.check_stage3
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from env import prompt as prompt_mod
from env.ast_check import check as ast_violations
from env.inputs import TRAIN_HIDDEN_SEEDS, VISIBLE_SEEDS, make_inputs
from env.reward import compute_references, score
from harness.cheaters import CHEATERS
from harness.policies import fence
from tasks.seed_tasks import SEED_TASKS


def main():
    ok = True

    # 1. AST false positives
    fps = {t["id"]: ast_violations(t["modern"]) for t in SEED_TASKS if ast_violations(t["modern"])}
    print(f"1. AST check on {len(SEED_TASKS)} correct migrations: "
          f"{'no false positives' if not fps else 'FALSE POSITIVES ' + json.dumps(fps)}")
    ok &= not fps

    # shared inputs and references
    vis = {t["id"]: [make_inputs(t, s) for s in VISIBLE_SEEDS] for t in SEED_TASKS}
    hid = {t["id"]: [make_inputs(t, s) for s in TRAIN_HIDDEN_SEEDS] for t in SEED_TASKS}
    with ThreadPoolExecutor(8) as pool:
        refs = dict(zip([t["id"] for t in SEED_TASKS],
                        pool.map(lambda t: compute_references(t, vis[t["id"]] + hid[t["id"]]), SEED_TASKS)))
    nv = len(VISIBLE_SEEDS)

    # 2. cheater suite
    jobs = [(name, t, ast_on) for name in CHEATERS for t in SEED_TASKS for ast_on in (True, False)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda j: score(j[1], fence(CHEATERS[j[0]](j[1])), hid[j[1]["id"]],
                                                refs[j[1]["id"]][nv:], check_ast=j[2]), jobs))
    table = {}
    for (name, t, ast_on), r in zip(jobs, results):
        row = table.setdefault(name, {"on": [], "off": []})
        row["on" if ast_on else "off"].append(r)
    print(f"\n2. Cheater suite: {len(CHEATERS)} cheaters x {len(SEED_TASKS)} tasks (must all score 0 with the AST check on)\n")
    print(f"   {'cheater':28s} {'max reward':>10s}  {'stopped by':30s} {'passes w/o AST check':>20s}")
    for name, row in table.items():
        max_on = max(r.reward for r in row["on"])
        reasons = Counter(r.reason for r in row["on"])
        stopped = ", ".join(f"{k} {v}" for k, v in reasons.most_common(3))
        passes_off = sum(r.reward == 1.0 for r in row["off"])
        flag = "" if max_on <= 0 else "   <-- CHEATER WINS"
        ok &= max_on <= 0
        print(f"   {name:28s} {max_on:10.1f}  {stopped:30s} {passes_off:>13d} / {len(row['off'])}{flag}")

    # 3. leakage audit
    from env.feedback import feedback
    leaks = []
    for t in SEED_TASKS:
        fb, _ = feedback(t, t["legacy"], vis[t["id"]], refs[t["id"]][:nv])
        msgs = prompt_mod.build(t, vis[t["id"]], refs[t["id"]][:nv], fb)     # audit WITH feedback
        text = "\n".join(m["content"] for m in msgs)
        for i, inp in enumerate(hid[t["id"]]):
            for name, val in inp.items():
                s = json.dumps(val, separators=(", ", ": "))
                if len(s) > 12 and s in text:          # short scalars can collide by chance
                    leaks.append(f"{t['id']}: hidden input {i} ({name}) appears in prompt")
        for i, ref in enumerate(refs[t["id"]][nv:]):
            s = json.dumps(ref, separators=(", ", ": "))
            if s in text and s not in [json.dumps(r, separators=(", ", ": ")) for r in refs[t["id"]][:nv]]:
                leaks.append(f"{t['id']}: hidden reference {i} appears in prompt")
        if "nonce" in text or t["modern"].strip() in text:
            leaks.append(f"{t['id']}: sandbox internals or the solution appear in prompt")
    print(f"\n3. Leakage audit over {len(SEED_TASKS)} prompts (with execution feedback): {'clean' if not leaks else 'LEAKS'}")
    for l in leaks:
        print("   " + l)
    ok &= not leaks

    print("\nGATE PART 1: PASS" if ok else "\nGATE PART 1: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
