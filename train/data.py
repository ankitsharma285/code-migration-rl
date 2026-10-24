"""Task data for training: inputs, legacy reference outputs (cached), and tokenized prompts."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from env import prompt as prompt_mod
from env.inputs import TRAIN_HIDDEN_SEEDS, VISIBLE_SEEDS, make_inputs
from env.reward import compute_references
from harness.evaluate import load_tasks

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "results" / "cache"


def chat_ids(tokenizer, messages):
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys") and "input_ids" in out:      # some transformers versions return a dict
        out = out["input_ids"]
    return list(out)


def prepare(split, hidden_seeds=TRAIN_HIDDEN_SEEDS, workers=8, limit=None):
    """Returns a list of dicts: task, hidden inputs, hidden references, and the prompt messages.
    Legacy references are expensive (one sandbox job per task), so they are cached per split hash."""
    tasks = load_tasks(split)[:limit] if limit else load_tasks(split)
    key = hashlib.sha256(json.dumps([tasks, hidden_seeds], sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"refs_{split if not split.endswith('.jsonl') else Path(split).stem}_{key}.json"
    if path.exists():
        refs = json.loads(path.read_text())
    else:
        def one(t):
            inputs = [make_inputs(t, s) for s in VISIBLE_SEEDS + list(hidden_seeds)]
            return t["id"], compute_references(t, inputs)
        with ThreadPoolExecutor(workers) as pool:
            refs = dict(pool.map(one, tasks))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(refs))
    nv = len(VISIBLE_SEEDS)
    items = []
    for t in tasks:
        vis = [make_inputs(t, s) for s in VISIBLE_SEEDS]
        hid = [make_inputs(t, s) for s in hidden_seeds]
        items.append({"task": t, "hidden_inputs": hid, "hidden_refs": refs[t["id"]][nv:],
                      "messages": prompt_mod.build(t, vis, refs[t["id"]][:nv])})
    return items
