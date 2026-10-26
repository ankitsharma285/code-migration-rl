"""v2 gate for the multi-turn environment (automatic execution). Run before any multi-turn training.

  1. Loop mechanics with scripted policies (expected stop reason, turn count, final reward):
       oracle      -> passes visible at turn 1, stops: visible_pass, reward 1
       legacy      -> never fixes anything: max_turns, reward 0
       two_step    -> legacy, then the correct migration: stops at turn 2, reward 1 (pass_by_turn 0 -> 1)
       no_code     -> prose only: stops after 2 code-less replies, reward = malformed penalty
  2. Cheaters inside episodes: every v1 cheater plus an ADAPTIVE one that switches trick according to
     the feedback it gets. Max final reward must be <= 0 on every task.
  3. Leakage audit on EVERY turn of every episode: no hidden input or hidden reference appears in
     anything the policy sees (prompts and feedback).
  4. Context budget: largest conversation in characters (vLLM is served with 8192 tokens).

Usage (repo root):  envs/legacy/bin/python -m harness.check_multiturn
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from env.episode import run_episode
from env.inputs import TRAIN_HIDDEN_SEEDS, VISIBLE_SEEDS, make_inputs
from env.reward import MALFORMED_PENALTY, compute_references
from harness.cheaters import CHEATERS
from harness.policies import fence
from tasks.seed_tasks import SEED_TASKS

TURNS = 3


def turn_of(messages):
    return 1 + sum(m["role"] == "assistant" for m in messages)


def scripted(fn):
    return lambda task: (lambda messages: fence(fn(task, turn_of(messages), messages)))


ADAPTIVE_TRICKS = ["unchanged_legacy", "suppress_warnings", "monkeypatch_removed_apis", "private_append",
                   "swallow_errors", "hardcode_visible_examples"]


def adaptive(task, turn, messages):
    """Picks the next trick after each failure: what a reward-hacking policy could learn from feedback."""
    return CHEATERS[ADAPTIVE_TRICKS[(turn - 1) % len(ADAPTIVE_TRICKS)]](task)


POLICIES = {
    "oracle": scripted(lambda t, k, m: t["modern"]),
    "legacy": scripted(lambda t, k, m: t["legacy"]),
    "two_step": scripted(lambda t, k, m: t["legacy"] if k == 1 else t["modern"]),
    "no_code": lambda task: (lambda messages: "I would replace the deprecated calls."),
}
EXPECT = {  # name -> (stop reason, turns, final reward)
    "oracle": ("visible_pass", 1, 1.0),
    "legacy": ("max_turns", TURNS, 0.0),
    "two_step": ("visible_pass", 2, 1.0),
    "no_code": ("no_code", 2, MALFORMED_PENALTY),
}


def main():
    ok = True
    vis = {t["id"]: [make_inputs(t, s) for s in VISIBLE_SEEDS] for t in SEED_TASKS}
    hid = {t["id"]: [make_inputs(t, s) for s in TRAIN_HIDDEN_SEEDS] for t in SEED_TASKS}
    with ThreadPoolExecutor(8) as pool:
        refs = dict(zip([t["id"] for t in SEED_TASKS],
                        pool.map(lambda t: compute_references(t, vis[t["id"]] + hid[t["id"]]), SEED_TASKS)))
    nv = len(VISIBLE_SEEDS)

    def episodes(make_gen):
        with ThreadPoolExecutor(8) as pool:
            return list(pool.map(lambda t: run_episode(t, make_gen(t), vis[t["id"]], refs[t["id"]][:nv],
                                                       hid[t["id"]], refs[t["id"]][nv:], TURNS), SEED_TASKS))

    all_eps = []
    print(f"1. Loop mechanics ({len(SEED_TASKS)} seed tasks, {TURNS} turns)")
    for name, make_gen in POLICIES.items():
        eps = episodes(make_gen)
        all_eps += eps
        want = EXPECT[name]
        got = [(e.stop, len(e.turns), e.final.reward) for e in eps]
        bad = [(t["id"], g) for t, g in zip(SEED_TASKS, got) if g != want]
        ok &= not bad
        print(f"   {name:10s} expect stop={want[0]} turns={want[1]} reward={want[2]:+.1f}: "
              f"{'OK' if not bad else f'{len(bad)} MISMATCHES e.g. {bad[:2]}'}")

    print(f"\n2. Cheaters inside {TURNS}-turn episodes (must all score <= 0)")
    cheaters = {n: scripted(lambda t, k, m, f=f: f(t)) for n, f in CHEATERS.items()}
    cheaters["adaptive (switches trick per feedback)"] = scripted(adaptive)
    for name, make_gen in cheaters.items():
        eps = episodes(make_gen)
        all_eps += eps
        best = max(e.final.reward for e in eps)
        vis_pass = sum(e.stop == "visible_pass" for e in eps)
        ok &= best <= 0
        print(f"   {name:40s} max reward {best:+.1f}   stopped on a visible pass: {vis_pass:2d}/{len(eps)}"
              f"{'   <-- CHEATER WINS' if best > 0 else ''}")

    leaks, longest = [], 0
    for e, t in zip(all_eps, SEED_TASKS * (len(all_eps) // len(SEED_TASKS))):
        # environment-written text only: system/user messages and feedback. Assistant messages and the
        # "got:" lines echo the POLICY's own output, which can equal a hidden reference by coincidence
        # (a constant-0 cheater "matches" a hidden reference of 0) without any hidden data leaking.
        env_text = [m["content"] for tr in e.turns for m in tr.messages if m["role"] != "assistant"]
        env_text += [tr.feedback for tr in e.turns]
        text = "\n".join(l for chunk in env_text for l in chunk.splitlines() if not l.lstrip().startswith("got:"))
        longest = max(longest, max((sum(len(m["content"]) for m in tr.messages) + len(tr.completion)
                                    for tr in e.turns), default=0))
        for i, inp in enumerate(hid[t["id"]]):
            for name, val in inp.items():
                s = json.dumps(val, separators=(", ", ": "))
                if len(s) > 12 and s in text:
                    leaks.append(f"{t['id']}: hidden input {i} ({name})")
        visible = [json.dumps(r, separators=(", ", ": ")) for r in refs[t["id"]][:nv]]
        for i, ref in enumerate(refs[t["id"]][nv:]):
            s = json.dumps(ref, separators=(", ", ": "))
            if s in text and s not in visible:
                leaks.append(f"{t['id']}: hidden reference {i}")
    leaks = sorted(set(leaks))
    print(f"\n3. Leakage audit over {len(all_eps)} episodes, every turn: {'clean' if not leaks else 'LEAKS'}")
    for l in leaks[:20]:
        print("   " + l)
    ok &= not leaks
    print(f"\n4. Longest conversation: {longest} characters (~{longest // 3} tokens at ~3 chars/token; limit 8192)")
    print("\nMULTI-TURN GATE: PASS" if ok else "\nMULTI-TURN GATE: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
