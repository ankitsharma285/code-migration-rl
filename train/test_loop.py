"""End-to-end test of the multi-turn training loop on CPU: tiny random model, byte-level toy tokenizer,
and scripted generators. Rewards and feedback come from the real sandbox, so generate -> run on visible
inputs -> feedback turn -> score -> advantages -> update -> log -> checkpoint -> resume all run for real.

Two scripted policies check the evaluation's feedback lift:
  "uses feedback": turn 1 correct with p = 0.5, later turns (after seeing an error) correct with p = 0.9
  "resamples":     every turn correct with p = 0.5, whatever the feedback said
The first must show a clearly positive lift, the second a lift near 0.

Run:  envs/train/bin/python -m train.test_loop
"""
import json
import random
import shutil
import tempfile
from pathlib import Path

from collections import Counter

import torch

from train.data import prepare
from train.test_grpo import tiny_model
from train.train_grpo import DEFAULTS, Trainer


class ToyTokenizer:
    """Bytes shifted by 3; ids 0/1/2 are pad/eos/bos."""
    pad_token_id, eos_token_id, vocab = 0, 1, 259

    eos_token = None

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 3 for i in ids if i >= 3).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "")
        # keep CPU prompts short but task-specific: the legacy code sits after the shared instructions, and two
        # tasks with the same error would share the tail (the scripted generator maps prompts to tasks)
        return self.encode(text[150:650]) if tokenize else text

    def save_pretrained(self, d):
        Path(d, "toy_tokenizer.txt").write_text("byte-level toy")


class ScriptedGenerator:
    """Turn 1: the oracle migration with probability p_first, else the legacy code. Later turns (the prompt
    extends a turn-1 prompt): the oracle with probability p_later."""
    def __init__(self, tok, items, p_first=0.5, p_later=0.9, seed=0):
        self.tok, self.rng, self.p1, self.p2 = tok, random.Random(seed), p_first, p_later
        self.first = {tuple(tok.apply_chat_template(it["messages"])): it["task"] for it in items}
        self.syncs, self.turns_seen = 0, Counter()

    def task_of(self, p):
        for key, task in self.first.items():
            if tuple(p[:len(key)]) == key:
                return task, len(p) == len(key)
        raise KeyError("prompt does not extend any turn-1 prompt")

    def generate(self, prompt_ids, n, temperature=1.0, top_p=1.0, seed=None, seeds=None):
        out = []
        for p in prompt_ids:
            task, first = self.task_of(p)
            self.turns_seen["first" if first else "later"] += 1
            group = []
            for _ in range(n):
                good = self.rng.random() < (self.p1 if first else self.p2)
                code = task["modern"] if good else task["legacy"]
                text = f"```python\n{code.strip()}\n```"
                group.append((self.tok.encode(text) + [self.tok.eos_token_id], text, True))
            out.append(group)
        return out

    def sync_weights(self, model):
        self.syncs += 1


def make_trainer(out, cfg, p_first=0.5, p_later=0.9, eval_slice=slice(12, 16)):
    tok = ToyTokenizer()
    items = prepare("seed", workers=4)
    model = tiny_model()
    model.resize_token_embeddings(tok.vocab)
    gen = ScriptedGenerator(tok, items, p_first=p_first, p_later=p_later)
    return Trainer(cfg, out, model, None, tok, gen, items[:12], items[eval_slice], "cpu"), gen


def main():
    out = Path(tempfile.mkdtemp(prefix="grpo_loop_"))
    cfg = dict(DEFAULTS, prompts_per_step=4, group_size=4, max_steps=3, beta=0.0, lr=1e-3, eval_every=2,
               ckpt_every=2, score_workers=4, warmup_steps=1, micro_tokens=4096, micro_rows=4, logit_chunk=256,
               eval_samples=4)
    try:
        tr, gen = make_trainer(out, cfg, p_first=0.2, p_later=0.3)     # low enough that groups stay mixed
        tr.run(log=lambda s: print("   ", s))
        lines = [json.loads(l) for l in open(out / "log.jsonl")]
        steps = [l for l in lines if "reward_mean" in l]
        evals = [l for l in lines if "eval_pass_at_1" in l]
        assert len(steps) == 3, f"expected 3 step records, got {len(steps)}"
        need = {"reward_mean", "pass_rate", "mixed_pass_group_frac", "turns_mean", "stops", "visible_pass_by_turn",
                "vis_pass_hidden_fail", "policy_tokens_mean", "env_tokens_mean",
                "generate_s", "feedback_s", "score_s", "train_s", "sync_s"}
        missing = need - set(steps[-1])
        assert not missing, f"log is missing {missing}"
        trained = [s for s in steps if s["trained_episodes"]]
        assert trained, "at least one step should have mixed groups and train"
        missing = {"kl", "entropy", "grad_norm", "micro_batches"} - set(trained[0])
        assert not missing, f"update stats missing {missing}"
        assert gen.turns_seen["later"] > 0, "failed episodes must get a second turn"
        assert all(s["env_tokens_mean"] > 0 for s in steps), "feedback tokens must be in the episodes (masked)"
        assert all(s["vis_pass_hidden_fail"] == 0 for s in steps), "oracle/legacy never pass visible but fail hidden"
        ev = evals[-1]
        assert ev["eval_pass_by_turn"][-1] >= ev["eval_pass_by_turn"][0], "pass can only grow with turns"
        assert gen.syncs >= 1, "weights must be synced after updates"
        assert (out / "ckpt-00002").exists() and (out / "resume" / "state.pt").exists()

        # token bookkeeping: the mask is 1 exactly on reply tokens
        eps = tr.episodes(tr.train_items[:2], 2, 1.0, seed=7)
        for e in eps:
            reply_tokens = sum(len(gen.tok.encode(t)) + 1 for t in e.texts)
            assert sum(e.cont_mask) == reply_tokens, "mask must cover the replies and nothing else"
            assert e.turns == 1 or len(e.cont_mask) > sum(e.cont_mask), "multi-turn episodes contain feedback"

        # resume: a fresh trainer picks up at step 3 with identical weights
        before = {k: v.clone() for k, v in tr.policy.state_dict().items()}
        tr2, _ = make_trainer(out, dict(cfg, max_steps=4), p_first=0.2, p_later=0.3)
        tr2.load_resume()
        assert tr2.step == 3, f"resume should restore step 3, got {tr2.step}"
        assert all(torch.equal(before[k], v) for k, v in tr2.policy.state_dict().items()), "weights restored"
        tr2.run(log=lambda s: None)
        assert tr2.step == 4

        # feedback lift: a feedback-using policy vs. a resampling one (more eval samples, fixed seeds)
        lifts = {}
        for name, p_later in (("uses feedback", 0.9), ("resamples", 0.5)):
            tr3, _ = make_trainer(out / name, dict(cfg, eval_samples=8), p_later=p_later, eval_slice=slice(0, 16))
            (out / name).mkdir(parents=True, exist_ok=True)
            lifts[name] = tr3.evaluate()["eval_feedback_lift"]
        print(f"    feedback lift: {lifts}")
        # both policies draw identical turn-1 samples (same scripted seed), so the difference is exact:
        # expected lifts are +0.12 (1 - .5*.1*.1 = 0.995 vs best-of-3 0.875) and 0 (0.875 vs 0.875)
        assert lifts["uses feedback"] - lifts["resamples"] > 0.05, "feedback use must raise the lift"
        assert abs(lifts["resamples"]) < 0.15, "pure resampling must show a lift near 0 (16 tasks: noisy)"
        print(f"\nLOOP TEST PASS: {len(steps)} steps, pass rates {[round(s['pass_rate'], 2) for s in steps]}, "
              f"turns {[round(s['turns_mean'], 2) for s in steps]}, eval by turn {ev['eval_pass_by_turn']}, resume OK")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
