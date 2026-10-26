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


def prepare(split, hidden_seeds=TRAIN_HIDDEN_SEEDS, workers=8, limit=None, feedback_prompt=True):
    """Returns a list of dicts: task, visible/hidden inputs and legacy references, the turn-1 feedback on the
    legacy code (v2), and the turn-1 prompt messages (with that feedback when feedback_prompt is on).
    References and feedback are expensive (sandbox jobs per task), so both are cached per split hash."""
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
    vis = {t["id"]: [make_inputs(t, s) for s in VISIBLE_SEEDS] for t in tasks}
    fb = {}
    if feedback_prompt:
        from env.feedback import feedback
        fpath = path.with_name(path.name.replace("refs_", "feedback_v1_"))
        if fpath.exists():
            fb = json.loads(fpath.read_text())
        else:
            with ThreadPoolExecutor(workers) as pool:
                fb = dict(zip([t["id"] for t in tasks], pool.map(
                    lambda t: feedback(t, t["legacy"], vis[t["id"]], refs[t["id"]][:nv])[0], tasks)))
            fpath.write_text(json.dumps(fb))
    items = []
    for t in tasks:
        hid = [make_inputs(t, s) for s in hidden_seeds]
        items.append({"task": t, "hidden_inputs": hid, "hidden_refs": refs[t["id"]][nv:],
                      "visible_inputs": vis[t["id"]], "visible_refs": refs[t["id"]][:nv],
                      "first_feedback": fb.get(t["id"]),
                      "messages": prompt_mod.build(t, vis[t["id"]], refs[t["id"]][:nv], fb.get(t["id"]))})
    return items
