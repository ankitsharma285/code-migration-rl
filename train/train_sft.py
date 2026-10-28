"""Supervised fine-tuning on filtered successes: the RFT baseline for Stage 8 (data: harness/rft_data.py).

Usage (on a GPU node, from the repo root):
  envs/train/bin/python -m train.train_sft --config configs/rft_1p5b.json --out results/rft_s1

Loss: mean negative log-likelihood of the completion tokens (the prompt is masked), the completion
ending with the EOS token exactly as in GRPO rollouts. Same precision recipe as GRPO: fp32 master
weights, bf16 autocast, gradient checkpointing, chunked LM head, sequences packed into micro-batches by
a token budget. An optimizer step covers `batch_size` sequences; the loss averages over every completion
token in that step (long completions are not down-weighted). Cosine schedule with linear warmup.

The recipe is fixed in advance (configs/rft_1p5b.json) and not tuned on the evaluation splits: choosing
epochs or lr by heldout scores would make the comparison with GRPO unfair in RFT's favour. The run saves
ckpt-epoch<k> after every epoch but the last and ckpt-final after the last; ckpt-final is the evaluated one.
"""
import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path

import torch

from harness import manifest
from train.data import chat_ids
from train.grpo import build_batch, completion_logprobs, pack

DEFAULTS = {
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "data": "results/rft/sft_data.jsonl",
    "seed": 1,
    "epochs": 2,
    "lr": 1e-5,
    "warmup_frac": 0.03,
    "weight_decay": 0.0,
    "batch_size": 32,                 # sequences per optimizer step
    "max_grad_norm": 1.0,
    "max_len": 8192,                  # prompt + completion; longer examples are dropped (counted in the log)
    "micro_tokens": 12288,
    "micro_rows": 8,
    "logit_chunk": 512,
    "gradient_checkpointing": True,
    "limit": None,                    # first N examples only (smoke runs)
}


def tokenize(tok, messages, completion):
    """(prompt_ids, completion_ids): the prompt exactly as GRPO tokenizes it (train/data.chat_ids), the
    completion's text tokens plus EOS (GRPO trains on the sampled EOS too)."""
    return chat_ids(tok, messages), tok.encode(completion, add_special_tokens=False) + [tok.eos_token_id]


def template_check(tok, messages, completion):
    """The prompt + completion tokens should render the same text as the chat template applied to the
    whole conversation (up to the template's trailing newline after EOS)."""
    p, c = tokenize(tok, messages, completion)
    ours = tok.decode(p + c)
    ref = tok.apply_chat_template(messages + [{"role": "assistant", "content": completion}], tokenize=False)
    return ref.startswith(ours) and len(ref) - len(ours) <= 2, ours[-200:], ref[-200:]


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def sft_step(model, optimizer, sequences, pad_id, max_tokens, max_rows, max_grad_norm, device,
             autocast_dtype=None, logit_chunk=None):
    """One optimizer step over `sequences` (list of (prompt_ids, completion_ids)). Returns stats."""
    lengths = [len(p) + len(c) for p, c in sequences]
    micro = [build_batch([sequences[i][0] for i in idx], [sequences[i][1] for i in idx], pad_id)
             for idx in pack(lengths, max_tokens, max_rows)]
    total = sum(b.completion_mask.sum().item() for b in micro) or 1.0
    ctx = (lambda: torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype)) if autocast_dtype \
        else (lambda: torch.autocast(device_type="cpu", enabled=False))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    nll = 0.0
    for b in micro:
        mb = b.to(device)
        with ctx():
            lp, _ = completion_logprobs(model, mb, chunk=logit_chunk)
            loss = -(lp * mb.completion_mask).sum() / total
        loss.backward()
        nll += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    optimizer.step()
    return {"nll": nll, "grad_norm": gn, "tokens": total, "micro_batches": len(micro)}


def main():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys: key=value")
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)

    rows = [json.loads(l) for l in open(cfg["data"])]
    if cfg["limit"]:
        rows = rows[: cfg["limit"]]
    record = manifest.build(cfg, [{"id": r["task_id"]} for r in rows])
    if record["git"]["dirty"] and not args.allow_dirty:
        raise SystemExit("Refusing to train from uncommitted code. Commit first, or pass --allow-dirty.\n"
                         + "\n".join(record["git"]["dirty_files"]))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists: pick a new --out or delete it")
    out.mkdir(parents=True)
    stats_path = Path(cfg["data"]).with_suffix(".stats.json")
    record["data_stats"] = json.loads(stats_path.read_text()) if stats_path.exists() else None
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "run.json").write_text(json.dumps(record, indent=1))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ok, ours, ref = template_check(tok, rows[0]["messages"], rows[0]["completion"])
    print(f"template check: {'ok' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"tokenized example does not match the chat template\nours: {ours!r}\nref:  {ref!r}")
    seqs, dropped = [], 0
    for r in rows:
        p, c = tokenize(tok, r["messages"], r["completion"])
        if len(p) + len(c) > cfg["max_len"]:
            dropped += 1
            continue
        seqs.append((p, c))
    print(f"{len(seqs)} examples ({dropped} longer than {cfg['max_len']} tokens dropped) from {cfg['data']}")

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).to(device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(len(seqs) / bs)
    total = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(cfg["warmup_frac"] * total))
    autocast = torch.bfloat16 if device == "cuda" else None
    print(f"{cfg['epochs']} epochs x {steps_per_epoch} steps = {total} steps, batch {bs}, lr {cfg['lr']}")

    logf = open(out / "log.jsonl", "a")
    step, t0 = 0, time.monotonic()
    for epoch in range(1, cfg["epochs"] + 1):
        order = list(range(len(seqs)))
        random.shuffle(order)
        for a in range(0, len(order), bs):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, cfg["lr"])
            st = sft_step(model, opt, [seqs[i] for i in order[a:a + bs]], tok.pad_token_id, cfg["micro_tokens"],
                          cfg["micro_rows"], cfg["max_grad_norm"], device, autocast, cfg["logit_chunk"])
            step += 1
            st.update(step=step, epoch=epoch, lr=opt.param_groups[0]["lr"], elapsed_s=round(time.monotonic() - t0))
            logf.write(json.dumps(st) + "\n")
            logf.flush()
            if step % 10 == 0 or step == total:
                print(f"step {step}/{total} epoch {epoch} nll {st['nll']:.4f} gn {st['grad_norm']:.2f} "
                      f"lr {st['lr']:.2e} {st['elapsed_s']}s", flush=True)
        d = out / ("ckpt-final" if epoch == cfg["epochs"] else f"ckpt-epoch{epoch}")
        model.save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
        print(f"saved {d}", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
