# MigrateRL: multi-turn RL with execution feedback for pandas 2 / NumPy 2 code migration

GRPO with execution-verified rewards trains Qwen2.5-Coder-1.5B-Instruct to migrate pandas 1.3 / NumPy 1.21
code to pandas 2.2 / NumPy 2.x. The model works as a multi-turn agent: each reply is run on example inputs
and the error, warning or output diff comes back as the next turn. The reward is binary and comes only from
running code: the migrated function must reproduce, on the new stack, exactly what the legacy function
returned on the old stack, on 10 hidden inputs.

Full write-up (method, controls, limitations): see the project report.

## Results

Evaluation hidden seeds, 2 training seeds pooled, 8 samples per task, 95% bootstrap CIs over tasks.
`heldout_id` = 200 unseen tasks of the training migration types (split by near-duplicate cluster);
`heldout_cat` = 138 tasks of 3 migration types never trained on.

| Arm | heldout_id final | heldout_id turn 1 | heldout_cat |
| --- | --- | --- | --- |
| Codemod (hand-written AST rules) | 59.5% [52.5, 66.5] | 59.5% | 0.0% |
| Base model, 3 turns | 17.6% [14.0, 21.4] | 9.4% | 7.2% |
| Base model, best of 3 samples | 16.1% [11.8, 20.7] | - | 10.0% |
| RFT (SFT on base successes), 1 attempt | 45.8% | 45.8% | 7.5% |
| Single-turn GRPO, 1 attempt | 79.3% [74.1, 84.4] | 79.3% | 2.9% |
| Multi-turn GRPO | 83.9% [79.0, 88.6] | 58.9% | 7.9% |
| **Multi-turn GRPO + turn penalty** | **83.6% [78.7, 88.2]** | **78.2%** | **12.5%** |

Paired differences (percentage points):

- turn penalty vs base, 3 turns: **+65.9 [+60.7, +71.0]**; vs codemod: +24.1 [+15.8, +32.3]
- single-turn GRPO vs RFT at a matched generation budget (77k samples): +33.5 [+28.6, +38.2]
- multi-turn GRPO with real vs placebo feedback ("your code did not pass"): +10.1 [+6.8, +13.6]
  (+3.1 [+1.0, +5.6] with the turn penalty)
- unseen types, turn penalty vs base best of 3: +2.5 [+0.3, +4.9] (weak; added after seeing the table)
- MBPP (general coding, greedy): 66.5% vs 66.1% base, +0.4 [-3.1, +3.9]

`python -m harness.final_table configs/final_table.json` rebuilds every cell and comparison above (and the
hard split) from the run directories.

## Repository layout

| Path | What it holds |
| --- | --- |
| `env/` | the environment: input generators, sandbox (`runner.py`, `child.py`), output comparison, AST check, reward, feedback, multi-turn episode loop |
| `tasks/` | task generation (`generate.py`), the filter that verifies every task (`filter_tasks.py`), split building, migration-type catalogue; frozen splits in `tasks/splits/` |
| `train/` | GRPO (`grpo.py`, `train_grpo.py`, `rollout.py`), in-process vLLM generation and weight sync, SFT for the RFT baseline (`train_sft.py`), unit tests |
| `harness/` | evaluation, codemod baseline, scripted cheaters and gates, rescoring, comparisons and the final table, MBPP, audits, profiling report |
| `configs/` | every training config used in the report, plus the final-table spec |
| `slurm/` | job scripts (one H100 per job) |
| `docs/` | dataset notes and the leakage audit |

## Setup

Three Python environments, deliberately separate: `envs/legacy` (pandas 1.3.5, NumPy 1.21.6) produces
reference outputs, `envs/modern` (pandas 2.2.3, NumPy 2.2.6) runs candidate code, and `envs/train`
(vLLM + PyTorch) serves and trains models. vLLM and torch pin their own NumPy, so sharing an environment
could silently change what counts as a correct migration.

```bash
bash setup_task_envs.sh        # envs/legacy and envs/modern from envs/*.lock (CPU, any Linux box)
bash setup_train_env.sh        # envs/train on a GPU node
envs/legacy/bin/python -m harness.check_multiturn   # gate: loop checks, leakage audit, filesystem canaries; every scripted cheater must score 0
```

The sandbox needs Linux with unprivileged user namespaces (`unshare -rmn`) for its network block and
filesystem lock; the gate refuses to pass without them. Training and evaluation refuse to run from uncommitted
code, so each result records the commit that produced it.

## Reproduce

**Without a GPU (minutes):** re-score stored model replies. Download the released results bundle into
`results/`, then:

```bash
envs/legacy/bin/python -m harness.reproduce results/final/tp_ck00150_mt3_heldout_id   # checks out the recorded commit, re-scores, compares
envs/legacy/bin/python -m harness.final_table configs/final_table.json              # the whole results table
```

**Retrain and evaluate one arm (one H100, ~2.5 h training):**

```bash
CONFIG=configs/mt_1p5b_tp.json OUT=results/mt_tp_s1 sbatch slurm/train_grpo.sbatch
MODEL=results/mt_tp_s1/ckpt-00150 TASKS=heldout_id SAMPLES=8 FEEDBACK=1 TURNS=3 SEEDS=eval:final \
  OUT=results/final/tp_ck00150_mt3_heldout_id sbatch slurm/eval_model.sbatch
```

Other arms: `configs/mt_1p5b.json` (multi-turn, no penalty), `configs/st_1p5b.json` (single-turn control),
`configs/rft_1p5b.json` with `harness/rft_data.py` and `slurm/train_sft.sbatch` (RFT). Second seeds use
the `_s2` configs (identical except `seed: 2`).

## Data

`tasks/splits/` holds the frozen splits with a manifest of SHA-256 hashes: train 1,762, heldout_id 200,
heldout_cat 138, hard_id 238, hard_cat 21. Tasks were written by Gemma 4 31B and kept only if the legacy code
runs, its behavior changes across versions, and a reference migration reproduces it; see
`docs/dataset_notes.md` for the pipeline and its known biases. MBPP (sanitized) is included for the
regression check under its original license (CC BY 4.0).

## Known limitations

One 1.5B model, one library upgrade, generated single-function tasks. On migration types never trained on,
no method clearly beats sampling the base model three times. The task filter removed the hardest behavior
changes from training and evaluation alike (e.g. `sum(level=0)` ordering, which the codemod handles and the
model never learns). RFT is a single round; iterated RFT is the stronger baseline and is not run here.

## License

MIT for the code (see `LICENSE`). Generated tasks are released for research use, subject to the terms of the
model that generated them.
