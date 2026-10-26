"""GRPO training loop: one GPU runs both the trainer and vLLM. v2: multi-turn episodes with automatic
execution feedback (train/rollout.py); turns=1 with feedback_prompt is the single-turn control run.

Usage (on a GPU node, from the repo root):
  envs/train/bin/python -m train.train_grpo --config configs/grpo_1p5b.json --out results/grpo_s1
  ... --resume results/grpo_s1                 # continue after a Slurm time limit

Each step:
  1. sample `prompts_per_step` training tasks; run `group_size` episodes each (up to `turns` turns:
     vLLM generates every live episode's next reply in one batch, the sandbox runs each reply on the
     visible inputs and the feedback becomes the next user turn)
  2. score each episode's LAST attempt on the hidden inputs (env/reward.py)
  3. group-normalized advantages; groups with identical rewards carry no signal and are dropped
  4. one GRPO update over every reply token of every turn (feedback tokens masked out), then copy the
     new weights into vLLM
  5. log one JSON line to <out>/log.jsonl; evaluate and checkpoint on schedule

Checkpoint evals (sampled, fixed seed) report: turn-1 pass rate, pass by turn, the feedback lift
(final multi-turn pass minus visible-selected best-of-`turns` single-turn attempts) and the
visible-pass/hidden-fail gap (a rise means the policy is learning to game the visible examples).

The task sandbox runs under envs/legacy/bin/python (the harness modules use only the standard library
in this process), so this file runs in envs/train with torch and vLLM.
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from harness import manifest
from harness.bestofn import visible_best_of_k_exact
from train.data import chat_ids, prepare
from train.grpo import group_advantages, grpo_step_packed, mixed_group_fraction
from train.rollout import TurnTemplate, rollout, template_consistent

DEFAULTS = {
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "seed": 1,
    "train_split": "train",
    "eval_split": "heldout_id",
    "eval_tasks": 64,
    "prompts_per_step": 32,
    "group_size": 8,
    "max_steps": 300,
    "lr": 1e-6,
    "warmup_steps": 10,
    "beta": 0.02,
    "clip_eps": 0.2,
    "ppo_epochs": 1,
    "micro_batch": 4,
    "max_grad_norm": 1.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,               # per turn
    "max_model_len": 8192,                # whole episode (prompt + every turn)
    "turns": 3,                           # 1 = single-turn (the control run)
    "feedback_prompt": True,              # turn 1 shows the legacy code's errors on the visible inputs
    "turn_penalty": 0.0,                  # subtracted per extra turn from a passing episode (reward pilot)
    "micro_tokens": 16384,                # token budget per micro-batch (rows x longest row)
    "micro_rows": 8,
    "logit_chunk": 1024,                  # LM-head positions per chunk (memory); 0 = off
    "eval_samples": 4,                    # episodes per eval task (>= turns, for best-of-`turns`)
    "eval_temperature": 1.0,
    "eval_seed": 4242,                    # same seed at every checkpoint: comparable evals
    "vllm_gpu_memory_utilization": 0.35,
    "drop_uniform_groups": True,
    "gradient_checkpointing": True,
    "score_workers": 16,
    "eval_every": 25,
    "ckpt_every": 50,
    "time_budget_min": 225,
}


class Trainer:
    def __init__(self, cfg, out, policy, ref, tokenizer, generator, train_items, eval_items, device):
        self.cfg, self.out, self.device = cfg, Path(out), device
        self.policy, self.ref, self.tok, self.gen = policy, ref, tokenizer, generator
        self.train_items, self.eval_items = train_items, eval_items
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / max(1, cfg["warmup_steps"])))
        self.step, self.rng = 0, random.Random(cfg["seed"])
        self.order, self.cursor = [], 0
        self.prompt_ids = {it["task"]["id"]: chat_ids(tokenizer, it["messages"])
                           for it in train_items + eval_items}
        self.autocast = torch.bfloat16 if device.startswith("cuda") else None
        self.tmpl = TurnTemplate(tokenizer)

    # -- data
    def next_items(self):
        n = self.cfg["prompts_per_step"]
        picked = []
        while len(picked) < n:
            if self.cursor >= len(self.order):                 # new epoch: reshuffle
                self.order = list(range(len(self.train_items)))
                self.rng.shuffle(self.order)
                self.cursor = 0
            picked.append(self.train_items[self.order[self.cursor]])
            self.cursor += 1
        return picked

    # -- episodes
    def episodes(self, items, n, temperature, seed, score_turns=False, timing=None):
        c = self.cfg
        return rollout(self.gen, self.tok, self.tmpl, items, n, c["turns"], temperature, c["top_p"], seed,
                       c["max_model_len"], c["max_new_tokens"], c["score_workers"], score_turns,
                       self.prompt_ids, timing)

    # -- one step
    def train_step(self):
        cfg, G, t = self.cfg, self.cfg["group_size"], {}
        items = self.next_items()
        eps = self.episodes(items, G, cfg["temperature"], cfg["seed"] * 100_000 + self.step, timing=t)
        passed = torch.tensor([e.reward == 1.0 for e in eps], dtype=torch.float)
        rewards = torch.tensor([e.reward - cfg["turn_penalty"] * (e.turns - 1) * (e.reward == 1.0) for e in eps])
        adv = group_advantages(rewards, G)

        keep = list(range(len(eps)))
        if cfg["drop_uniform_groups"]:
            groups = rewards.view(-1, G)
            informative = (groups.max(1).values > groups.min(1).values).nonzero().flatten().tolist()
            keep = [g * G + k for g in informative for k in range(G)]

        stats = {}
        t0 = time.monotonic()
        if keep:
            seqs = [(eps[i].prompt_ids, eps[i].cont_ids, eps[i].cont_mask) for i in keep]
            stats = grpo_step_packed(self.policy, self.opt, seqs, adv[keep], self.tok.pad_token_id,
                                     ref_model=self.ref, beta=cfg["beta"], clip_eps=cfg["clip_eps"],
                                     max_tokens=cfg["micro_tokens"], max_rows=cfg["micro_rows"],
                                     ppo_epochs=cfg["ppo_epochs"], max_grad_norm=cfg["max_grad_norm"],
                                     device=self.device, autocast_dtype=self.autocast,
                                     logit_chunk=cfg["logit_chunk"] or None)
            self.sched.step()
        t["train_s"] = time.monotonic() - t0

        t0 = time.monotonic()
        if keep:
            self.gen.sync_weights(self.policy)
        t["sync_s"] = time.monotonic() - t0
        self.step += 1

        n, T = max(len(eps), 1), cfg["turns"]
        reasons = Counter(e.final.reason if e.final else "context" for e in eps)
        policy_tok = [sum(e.cont_mask) for e in eps]
        ep_len = [len(e.prompt_ids) + len(e.cont_ids) for e in eps]
        replies = [d for e in eps for d in e.finished]
        rec = {"step": self.step, "lr": self.sched.get_last_lr()[0],
               "reward_mean": rewards.mean().item(), "pass_rate": passed.mean().item(),
               "mixed_group_frac": mixed_group_fraction(rewards, G),
               # groups with BOTH a pass and a non-pass: the real task signal
               "mixed_pass_group_frac": mixed_group_fraction(passed, G),
               "trained_episodes": len(keep),
               "turns_mean": sum(e.turns for e in eps) / n,
               "stops": dict(Counter(e.stop for e in eps)),
               # cumulative share of episodes whose code passed the VISIBLE examples by turn k
               "visible_pass_by_turn": [round(sum(any(e.visible_ok[:k]) for e in eps) / n, 4) for k in range(1, T + 1)],
               "vis_pass_hidden_fail": sum(e.stop == "visible_pass" and e.reward != 1.0 for e in eps) / n,
               "reasons": dict(reasons), "ast_violations": reasons.get("ast", 0),
               "policy_tokens_mean": sum(policy_tok) / n,
               "env_tokens_mean": sum(len(e.cont_mask) - p for e, p in zip(eps, policy_tok)) / n,
               "episode_tokens_mean": sum(ep_len) / n, "episode_tokens_max": max(ep_len, default=0),
               "truncated_frac": sum(not d for d in replies) / max(len(replies), 1),
               **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
               **{k: round(v, 2) for k, v in t.items()}}
        if self.step == 1:          # once: are turn boundaries tokenized exactly as the chat template does?
            checks = [c for c in (template_consistent(self.tok, e) for e in eps[:64]) if c is not None]
            rec["template_consistent"] = (sum(checks) / len(checks)) if checks else None
        if torch.cuda.is_available():
            rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
        return rec

    # -- evaluation: sampled episodes on a fixed held-out sample, every turn scored on hidden inputs
    def evaluate(self):
        c, T, n = self.cfg, self.cfg["turns"], self.cfg["eval_samples"]
        eps = self.episodes(self.eval_items, n, c["eval_temperature"], c["eval_seed"], score_turns=True)
        N = max(len(eps), 1)
        at = lambda e, k: bool(e.hidden) and e.hidden[min(k, len(e.hidden)) - 1].reward == 1.0
        by_turn = [sum(at(e, k) for e in eps) / N for k in range(1, T + 1)]
        # best-of-T from each task's TURN-1 attempts: the same model resampled, no feedback used
        bo = []
        for g in range(0, len(eps), n):
            grp = [(e.visible_ok[0], e.hidden[0].reward == 1.0) for e in eps[g:g + n] if e.hidden]
            if len(grp) >= T:
                bo.append(visible_best_of_k_exact(grp, T))
        best_of = sum(bo) / max(len(bo), 1)
        return {"step": self.step, "eval_pass_at_1": by_turn[-1], "eval_turn1_pass": by_turn[0],
                "eval_pass_by_turn": [round(x, 4) for x in by_turn],
                f"eval_visible_best_of_{T}": round(best_of, 4),
                "eval_feedback_lift": round(by_turn[-1] - best_of, 4),
                "eval_vis_pass_hidden_fail": sum(e.stop == "visible_pass" and e.reward != 1.0 for e in eps) / N,
                "eval_turns_mean": sum(e.turns for e in eps) / N,
                "eval_reasons": dict(Counter(e.final.reason if e.final else "context" for e in eps))}

    # -- checkpoints
    def save(self, tag):
        d = self.out / tag
        d.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(d, safe_serialization=True)
        self.tok.save_pretrained(d)

    def save_resume(self):
        d = self.out / "resume"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(), "step": self.step, "rng": self.rng.getstate(),
                    "order": self.order, "cursor": self.cursor}, d / "state.pt.tmp")
        os.replace(d / "state.pt.tmp", d / "state.pt")             # atomic: a kill mid-save can't corrupt it

    def load_resume(self):
        s = torch.load(self.out / "resume" / "state.pt", map_location="cpu", weights_only=False)
        self.policy.load_state_dict(s["policy"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.step, self.order, self.cursor = s["step"], s["order"], s["cursor"]
        self.rng.setstate(s["rng"])
        self.gen.sync_weights(self.policy)

    # -- main loop
    def run(self, log=print):
        start, cfg = time.monotonic(), self.cfg
        logf = open(self.out / "log.jsonl", "a")
        while self.step < cfg["max_steps"]:
            rec = self.train_step()
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {rec['step']:4d}  reward {rec['reward_mean']:+.3f}  pass {rec['pass_rate']:.3f}  "
                f"pass-mixed {rec['mixed_pass_group_frac']:.2f}  turns {rec['turns_mean']:.2f}  "
                f"kl {rec.get('kl', 0):.4f}  vis-pass/hid-fail {rec['vis_pass_hidden_fail']:.3f}  "
                f"gen {rec['generate_s']:.0f}s env {rec['feedback_s']:.0f}s score {rec['score_s']:.0f}s "
                f"train {rec['train_s']:.0f}s")
            if self.eval_items and (self.step % cfg["eval_every"] == 0 or self.step == cfg["max_steps"]):
                ev = self.evaluate()
                logf.write(json.dumps(ev) + "\n"); logf.flush()
                log(f"eval  step {ev['step']}  turn-1 {ev['eval_turn1_pass']:.3f}  by turn {ev['eval_pass_by_turn']}  "
                    f"feedback lift {ev['eval_feedback_lift']:+.3f}  vis-pass/hid-fail {ev['eval_vis_pass_hidden_fail']:.3f}")
            if self.step % cfg["ckpt_every"] == 0 or self.step == cfg["max_steps"]:
                self.save(f"ckpt-{self.step:05d}")
                self.save_resume()
            if (time.monotonic() - start) / 60 > cfg["time_budget_min"]:
                self.save_resume()
                log(f"time budget reached at step {self.step}; resume with --resume {self.out}")
                break
        logf.close()


def build(cfg, out, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train.generation import VLLMGenerator
    torch.manual_seed(cfg["seed"])
    # vLLM first: it sizes its memory pool from what is free when it starts
    gen = VLLMGenerator(cfg["model"], max_model_len=cfg["max_model_len"], max_new_tokens=cfg["max_new_tokens"],
                        gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"], seed=cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # fp32 master weights: at lr ~1e-6 a bf16 weight would round most updates away
    policy = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).to(device)
    if cfg["gradient_checkpointing"]:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    ref = None
    if cfg["beta"] > 0:
        ref = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16).to(device).eval()
        ref.requires_grad_(False)
    train_items = prepare(cfg["train_split"], workers=cfg["score_workers"], feedback_prompt=cfg["feedback_prompt"])
    eval_items = prepare(cfg["eval_split"], workers=cfg["score_workers"],
                         feedback_prompt=cfg["feedback_prompt"])[: cfg["eval_tasks"]]
    return Trainer(cfg, out, policy, ref, tok, gen, train_items, eval_items, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="continue the run in --out")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys: key=value")
    args = ap.parse_args()

    out = Path(args.out)
    if args.resume:
        cfg = json.loads((out / "config.json").read_text())
    else:
        cfg = dict(DEFAULTS)
        if args.config:
            cfg.update(json.loads(Path(args.config).read_text()))
        for kv in args.set:
            k, v = kv.split("=", 1)
            cfg[k] = type(DEFAULTS[k])(json.loads(v)) if k in DEFAULTS else json.loads(v)
        from harness.evaluate import load_tasks
        record = manifest.build(cfg, load_tasks(cfg["train_split"]))
        if record["git"]["dirty"] and not args.allow_dirty:      # checked BEFORE creating --out
            raise SystemExit("Refusing to train from uncommitted code (results would not be reproducible). "
                             "Commit first, or pass --allow-dirty.\n" + "\n".join(record["git"]["dirty_files"]))
        if out.exists():
            raise SystemExit(f"{out} already exists: pick a new --out, delete it, or pass --resume to continue it")
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
        (out / "run.json").write_text(json.dumps(record, indent=1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = build(cfg, out, device)
    if args.resume:
        trainer.load_resume()
        print(f"resumed at step {trainer.step}")
    trainer.run()


if __name__ == "__main__":
    main()
