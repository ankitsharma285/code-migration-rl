"""GRPO training loop: one GPU runs both the trainer and vLLM.

Usage (on a GPU node, from the repo root):
  envs/train/bin/python -m train.train_grpo --config configs/grpo_1p5b.json --out results/grpo_s1
  ... --resume results/grpo_s1                 # continue after a Slurm time limit

Each step:
  1. sample `prompts_per_step` training tasks; generate `group_size` completions each (vLLM)
  2. score every completion in the sandbox (env/reward.py), in parallel on the CPU cores
  3. group-normalized advantages; groups with identical rewards carry no signal and are dropped
  4. one GRPO update (train/grpo.py), then copy the new weights into vLLM
  5. log one JSON line to <out>/log.jsonl; evaluate and checkpoint on schedule

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

from env.reward import score
from harness import manifest
from train.data import chat_ids, prepare
from train.grpo import build_batch, group_advantages, grpo_step, mixed_group_fraction

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
    "max_new_tokens": 1024,
    "max_model_len": 4096,
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

    # -- scoring
    def score_all(self, items, samples):
        jobs = [(it, s) for it, group in zip(items, samples) for s in group]
        with ThreadPoolExecutor(self.cfg["score_workers"]) as pool:
            return list(pool.map(lambda j: score(j[0]["task"], j[1][1], j[0]["hidden_inputs"], j[0]["hidden_refs"]),
                                 jobs))

    # -- one step
    def train_step(self):
        cfg, G, t = self.cfg, self.cfg["group_size"], {}
        items = self.next_items()
        t0 = time.monotonic()
        samples = self.gen.generate([self.prompt_ids[it["task"]["id"]] for it in items], n=G,
                                    temperature=cfg["temperature"], top_p=cfg["top_p"],
                                    seed=cfg["seed"] * 100_000 + self.step)
        t["generate_s"] = time.monotonic() - t0

        t0 = time.monotonic()
        results = self.score_all(items, samples)
        t["score_s"] = time.monotonic() - t0
        rewards = torch.tensor([r.reward for r in results])
        adv = group_advantages(rewards, G)

        keep = list(range(len(items) * G))
        if cfg["drop_uniform_groups"]:
            groups = rewards.view(-1, G)
            informative = (groups.max(1).values > groups.min(1).values).nonzero().flatten().tolist()
            keep = [g * G + k for g in informative for k in range(G)]

        stats = {}
        t0 = time.monotonic()
        if keep:
            prompts = [self.prompt_ids[items[i // G]["task"]["id"]] for i in keep]
            comps = [samples[i // G][i % G][0] for i in keep]
            batch = build_batch(prompts, comps, self.tok.pad_token_id)
            stats = grpo_step(self.policy, self.opt, batch, adv[keep], ref_model=self.ref, beta=cfg["beta"],
                              clip_eps=cfg["clip_eps"], micro_batch=cfg["micro_batch"],
                              ppo_epochs=cfg["ppo_epochs"], max_grad_norm=cfg["max_grad_norm"],
                              device=self.device, autocast_dtype=self.autocast)
            self.sched.step()
        t["train_s"] = time.monotonic() - t0

        t0 = time.monotonic()
        if keep:
            self.gen.sync_weights(self.policy)
        t["sync_s"] = time.monotonic() - t0
        self.step += 1

        flat = [s for group in samples for s in group]
        reasons = Counter(r.reason for r in results)
        rec = {"step": self.step, "lr": self.sched.get_last_lr()[0],
               "reward_mean": rewards.mean().item(), "pass_rate": (rewards == 1.0).float().mean().item(),
               "mixed_group_frac": mixed_group_fraction(rewards, G), "trained_sequences": len(keep),
               "reasons": dict(reasons), "ast_violations": reasons.get("ast", 0),
               "completion_len_mean": sum(len(s[0]) for s in flat) / max(len(flat), 1),
               "truncated_frac": sum(not s[2] for s in flat) / max(len(flat), 1),
               **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
               **{k: round(v, 2) for k, v in t.items()},
               "sandbox_wall_median_s": sorted(r.wall_s for r in results)[len(results) // 2]}
        if torch.cuda.is_available():
            rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
        return rec

    # -- evaluation: greedy pass@1 on a fixed held-out sample
    def evaluate(self):
        items = self.eval_items
        samples = self.gen.generate([self.prompt_ids[it["task"]["id"]] for it in items], n=1, temperature=0.0)
        results = self.score_all(items, samples)
        return {"step": self.step, "eval_pass_at_1": sum(r.reward == 1.0 for r in results) / len(results),
                "eval_reasons": dict(Counter(r.reason for r in results))}

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
                f"mixed {rec['mixed_group_frac']:.2f}  kl {rec.get('kl', 0):.4f}  "
                f"gen {rec['generate_s']:.0f}s score {rec['score_s']:.0f}s train {rec['train_s']:.0f}s")
            if self.eval_items and (self.step % cfg["eval_every"] == 0 or self.step == cfg["max_steps"]):
                ev = self.evaluate()
                logf.write(json.dumps(ev) + "\n"); logf.flush()
                log(f"eval  step {ev['step']}  pass@1 {ev['eval_pass_at_1']:.3f}")
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
    train_items = prepare(cfg["train_split"], workers=cfg["score_workers"])
    eval_items = prepare(cfg["eval_split"], workers=cfg["score_workers"])[: cfg["eval_tasks"]]
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
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
        from harness.evaluate import load_tasks
        record = manifest.build(cfg, load_tasks(cfg["train_split"]))
        if record["git"]["dirty"] and not args.allow_dirty:
            raise SystemExit("Refusing to train from uncommitted code (results would not be reproducible). "
                             "Commit first, or pass --allow-dirty.")
        (out / "run.json").write_text(json.dumps(record, indent=1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = build(cfg, out, device)
    if args.resume:
        trainer.load_resume()
        print(f"resumed at step {trainer.step}")
    trainer.run()


if __name__ == "__main__":
    main()
