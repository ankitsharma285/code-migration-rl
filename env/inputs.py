"""Input generation, done once in the parent so both runtimes see identical data.

A task's `gen` source defines gen(rng) -> kwargs with rng = random.Random(seed); values are plain
JSON data, with tagged dicts that env/child.py turns into pandas objects. Python's `random` is
used instead of NumPy's Generator because NumPy does not guarantee identical streams across
versions, and both runtimes share the same Python 3.10.
"""
import random

VISIBLE_SEEDS = [0, 1]                    # shown in the prompt as examples
TRAIN_HIDDEN_SEEDS = list(range(100, 110))  # scored during training


def make_inputs(task, seed):
    ns = {}
    exec(task["gen"], ns)
    return ns["gen"](random.Random(seed))


def eval_seeds(round_id, n=10):
    """Fresh hidden seeds for evaluation, disjoint from training seeds (re-randomized per eval round)."""
    rng = random.Random(f"eval-{round_id}")
    return rng.sample(range(10_000, 10_000_000), n)
