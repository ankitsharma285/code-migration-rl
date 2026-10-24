"""End-to-end test of the training loop on CPU: tiny random model, byte-level toy tokenizer, and a
scripted generator that answers each seed task with the correct migration or the unchanged legacy
code. Rewards come from the real sandbox, so generate -> score -> advantages -> update -> log ->
checkpoint -> resume all run for real.

Run:  envs/train/bin/python -m train.test_loop
"""
import json
import random
import shutil
import tempfile
from pathlib import Path

import torch

from train.data import prepare
from train.test_grpo import tiny_model
from train.train_grpo import DEFAULTS, Trainer


class ToyTokenizer:
    """Bytes shifted by 3; ids 0/1/2 are pad/eos/bos."""
    pad_token_id, eos_token_id, vocab = 0, 1, 259

    def encode(self, text):
        return [b + 3 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 3 for i in ids if i >= 3).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return self.encode(text[-400:])                 # keep CPU sequences short

    def save_pretrained(self, d):
        Path(d, "toy_tokenizer.txt").write_text("byte-level toy")


class ScriptedGenerator:
    """Answers with the oracle migration with probability p_correct, else the legacy code."""
    def __init__(self, tok, items, p_correct=0.5, seed=0):
        self.tok, self.rng, self.p = tok, random.Random(seed), p_correct
        self.by_prompt = {tuple(tok.apply_chat_template(it["messages"])): it["task"] for it in items}
        self.syncs = 0

    def generate(self, prompt_ids, n, temperature=1.0, top_p=1.0, seed=None):
        out = []
        for p in prompt_ids:
            task = self.by_prompt[tuple(p)]
            group = []
            for _ in range(n):
                code = task["modern"] if (temperature == 0 or self.rng.random() < self.p) else task["legacy"]
                text = f"```python\n{code.strip()}\n```"
                ids = self.tok.encode(text)[:120] + [self.tok.eos_token_id]   # truncated ids: CPU speed
                group.append((ids, text, True))
            out.append(group)
        return out

    def sync_weights(self, model):
        self.syncs += 1


def make_trainer(out, cfg):
    tok = ToyTokenizer()
    items = prepare("seed", workers=2)
    model = tiny_model()
    model.resize_token_embeddings(tok.vocab)
    gen = ScriptedGenerator(tok, items)
    return Trainer(cfg, out, model, None, tok, gen, items[:12], items[12:16], "cpu"), gen


def main():
    out = Path(tempfile.mkdtemp(prefix="grpo_loop_"))
    cfg = dict(DEFAULTS, prompts_per_step=4, group_size=4, max_steps=3, micro_batch=4, beta=0.0, lr=1e-3,
               eval_every=2, ckpt_every=2, score_workers=2, warmup_steps=1)
    try:
        tr, gen = make_trainer(out, cfg)
        tr.run(log=lambda s: print("   ", s))
        lines = [json.loads(l) for l in open(out / "log.jsonl")]
        steps = [l for l in lines if "reward_mean" in l]
        evals = [l for l in lines if "eval_pass_at_1" in l]
        assert len(steps) == 3, f"expected 3 step records, got {len(steps)}"
        need = {"reward_mean", "pass_rate", "mixed_group_frac", "kl", "entropy", "clip_frac", "grad_norm",
                "completion_len_mean", "ast_violations", "generate_s", "score_s", "train_s", "sync_s"}
        missing = need - set(steps[-1])
        assert not missing, f"log is missing {missing}"
        assert any(0 < s["pass_rate"] < 1 for s in steps), "scripted mix should give mixed rewards"
        assert evals and evals[-1]["eval_pass_at_1"] == 1.0, "greedy scripted answers are the oracle: eval = 1.0"
        assert gen.syncs >= 1, "weights must be synced after updates"
        assert (out / "ckpt-00002").exists() and (out / "resume" / "state.pt").exists()

        # resume: a fresh trainer picks up at step 3 with identical weights
        before = {k: v.clone() for k, v in tr.policy.state_dict().items()}
        tr2, _ = make_trainer(out, dict(cfg, max_steps=4))
        tr2.load_resume()
        assert tr2.step == 3, f"resume should restore step 3, got {tr2.step}"
        assert all(torch.equal(before[k], v) for k, v in tr2.policy.state_dict().items()), "weights restored"
        tr2.run(log=lambda s: None)
        assert tr2.step == 4
        print(f"\nLOOP TEST PASS: {len(steps)} steps, pass rates {[round(s['pass_rate'], 2) for s in steps]}, "
              f"eval {evals[-1]['eval_pass_at_1']}, resume OK")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
