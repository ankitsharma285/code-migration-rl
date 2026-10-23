# Leakage and reward-hacking audit (Stage 3)

**Claim.** A policy cannot earn reward 1 on a task without producing a migration that behaves like the legacy code on inputs it has never seen.

## What the policy observes

The prompt (`env/prompt.py`) contains only the legacy function, its name, and two visible examples: inputs from the visible seeds and the legacy outputs on them. Scoring uses ten different hidden inputs and their legacy outputs. Neither appears in the prompt, and v1 returns no tool output to the policy. `harness/check_stage3.py` checks every prompt: no hidden input, hidden reference output, sandbox internal (nonce), or reference solution appears in it. Evaluation uses fresh hidden seeds per round (`env/inputs.eval_seeds`), disjoint from training seeds, so memorizing training inputs does not transfer.

## How the reward is defended

Each layer closes specific routes, verified by 13 scripted cheaters that all score 0 on all 20 seed tasks:

| Layer | Closes | Cheaters stopped |
|---|---|---|
| Hidden, re-randomized inputs | Hardcoding, constant or echoed outputs | `constant_output`, `echo_first_argument`, `hardcode_visible_examples` |
| AST check (`env/ast_check.py`) | Warning suppression, private APIs, `getattr` tricks, monkeypatching removed APIs back in, broad `except`, global option changes, imports outside an allow-list (`sys`, `gc`, `os`, ...) | `suppress_warnings`, `monkeypatch_removed_apis`, `clear_recorded_warnings`, `swallow_errors`, `global_option_change`, `copy_on_write_toggle`, and the `_append` variants where they reach `_append` |
| Sandbox (`env/runner.py`, `env/child.py`) | No network; memory, CPU, and wall limits; a per-job nonce rejects forged result output; in modern runs the legacy env is hidden behind an empty mount, so the old pandas cannot be loaded from disk | `load_legacy_pandas` (twice over), forged output |
| Runtime behavior check | Unchanged legacy code: crashes, deprecation warnings, or silent output changes | `unchanged_legacy` |

## Defense in depth

With the AST check switched off, four cheaters would have scored on some tasks: `monkeypatch_removed_apis` (3/20), `clear_recorded_warnings` (3/20, clearing the sandbox's warning record through `gc`), and `private_append` / `getattr_private_append` (1/20 each). `load_legacy_pandas` passed 13/20 until the legacy env was hidden in modern runs; now it fails even without the AST check. The AST check is therefore load-bearing for the first group, and the sandbox independently covers the most dangerous route.

## Known limits

- The AST check is an allow-list over syntax. Library functions that execute strings (e.g. `pd.eval`) are a residual route. Training-time logging counts AST violations per step, and a red-team run (Stage 6, day 7) reads rewarded rollouts by hand.
- Results depend on the pinned envs. `run.json` stores package freezes, and `harness/reproduce.py` warns when they differ.
