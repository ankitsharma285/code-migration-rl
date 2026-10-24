"""GRPO core, written from scratch: advantages, token log-probs, the clipped surrogate with KL, and one
optimizer step. Pure torch; generation (vLLM) and rewards (env/) are separate modules.

Notation, per prompt q with a group of G sampled completions o_1..o_G and rewards r_1..r_G:
  advantage   A_i = (r_i - mean(r)) / (std(r) + eps)          same A_i for every token of o_i
  ratio       rho_t = exp(logp_theta(o_t) - logp_old(o_t))
  surrogate   L_clip = -min(rho_t * A, clip(rho_t, 1-eps_c, 1+eps_c) * A)
  KL          k3 estimator of KL(pi_theta || pi_ref): exp(lr) - lr - 1, lr = logp_ref - logp_theta
  loss        sum over completion tokens of (L_clip + beta * KL), divided by the number of completion
              tokens in the whole batch (token-level averaging: long and short completions weigh
              each token equally, and gradient accumulation gives the same result as one big batch)

Sequence layout (makes the completion span start at the same position in every row):
  [left padding][prompt tokens][completion tokens][right padding]
  The model sees attention_mask and explicit position_ids, so padding never shifts positions.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- advantages

def group_advantages(rewards, group_size, eps=1e-4, scale_by_std=True):
    """rewards: 1-D tensor of length B*G, groups contiguous. Returns advantages of the same shape.
    A group whose rewards are all equal gets advantage 0 for every member (no learning signal)."""
    r = rewards.view(-1, group_size).float()
    centered = r - r.mean(dim=1, keepdim=True)
    if scale_by_std:
        std = r.std(dim=1, keepdim=True, unbiased=False)
        centered = centered / (std + eps)
    centered = torch.where(r.std(dim=1, keepdim=True, unbiased=False) > 0, centered, torch.zeros_like(centered))
    return centered.view(-1)


def mixed_group_fraction(rewards, group_size):
    """Fraction of groups with any reward variance: the only groups that produce gradient."""
    r = rewards.view(-1, group_size)
    return (r.max(dim=1).values > r.min(dim=1).values).float().mean().item()


# ---------------------------------------------------------------- batching

@dataclass
class Batch:
    input_ids: torch.Tensor        # [N, P + C]
    attention_mask: torch.Tensor   # [N, P + C]
    position_ids: torch.Tensor     # [N, P + C]
    completion_mask: torch.Tensor  # [N, C]  1 on real completion tokens (incl. EOS), 0 on padding
    prompt_len: int                # P: completion tokens occupy columns P .. P+C-1

    def to(self, device):
        return Batch(self.input_ids.to(device), self.attention_mask.to(device), self.position_ids.to(device),
                     self.completion_mask.to(device), self.prompt_len)

    def rows(self, idx):
        return Batch(self.input_ids[idx], self.attention_mask[idx], self.position_ids[idx],
                     self.completion_mask[idx], self.prompt_len)


def build_batch(prompt_ids, completion_ids, pad_id):
    """prompt_ids, completion_ids: lists of token-id lists (completion i belongs to prompt i)."""
    P = max(len(p) for p in prompt_ids)
    C = max(1, max(len(c) for c in completion_ids))
    n = len(prompt_ids)
    ids = torch.full((n, P + C), pad_id, dtype=torch.long)
    attn = torch.zeros((n, P + C), dtype=torch.long)
    cmask = torch.zeros((n, C), dtype=torch.float)
    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        ids[i, P - len(p):P] = torch.tensor(p, dtype=torch.long)
        attn[i, P - len(p):P] = 1
        if c:
            ids[i, P:P + len(c)] = torch.tensor(c, dtype=torch.long)
            attn[i, P:P + len(c)] = 1
            cmask[i, :len(c)] = 1.0
    pos = (attn.cumsum(dim=1) - 1).clamp(min=0)
    return Batch(ids, attn, pos, cmask, P)


# ---------------------------------------------------------------- log-probs

def completion_logprobs(model, batch, with_entropy=False):
    """Log-prob of each completion token under `model`: [N, C]. Optionally the per-token entropy.
    Only the completion span's logits are materialized for the log-softmax (vocab is ~150k)."""
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                position_ids=batch.position_ids, use_cache=False)
    P = batch.prompt_len
    logits = out.logits[:, P - 1:-1, :].float()          # position t predicts token t+1
    targets = batch.input_ids[:, P:]
    logZ = torch.logsumexp(logits, dim=-1)
    logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logZ
    if not with_entropy:
        return logp, None
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        entropy = logZ - (probs * logits).sum(-1)
    return logp, entropy


# ---------------------------------------------------------------- loss

def grpo_loss(logp, old_logp, ref_logp, advantages, completion_mask, total_tokens, clip_eps=0.2, beta=0.0):
    """Per-token clipped surrogate + beta * KL, summed and divided by `total_tokens` (all completion
    tokens in the full batch, so micro-batches can be accumulated). Returns (loss, stats dict)."""
    adv = advantages.unsqueeze(1)                                   # [N, 1] broadcast over tokens
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    per_token = -torch.minimum(unclipped, clipped)
    kl = torch.zeros_like(logp)
    if beta > 0 and ref_logp is not None:
        lr = ref_logp - logp
        kl = torch.exp(lr) - lr - 1                                 # k3: >= 0, unbiased for KL(theta||ref)
        per_token = per_token + beta * kl
    m = completion_mask
    loss = (per_token * m).sum() / total_tokens
    with torch.no_grad():
        n = m.sum().clamp(min=1)
        is_clipped = ((ratio < 1 - clip_eps) & (adv < 0)) | ((ratio > 1 + clip_eps) & (adv > 0))
        stats = {"kl_sum": (kl * m).sum().item(), "clip_sum": (is_clipped.float() * m).sum().item(),
                 "ratio_max": (ratio * m).max().item(), "tokens": n.item()}
    return loss, stats


# ---------------------------------------------------------------- one update

def grpo_step(model, optimizer, batch, advantages, ref_model=None, beta=0.0, clip_eps=0.2,
              micro_batch=4, ppo_epochs=1, max_grad_norm=1.0, device="cpu", autocast_dtype=None):
    """Old log-probs are computed once (no grad) from the current weights, then `ppo_epochs` passes
    over the batch in micro-batches with gradient accumulation. With ppo_epochs=1 the ratio is 1 and
    clipping is inactive; clipping matters only for the later passes."""
    N = batch.input_ids.shape[0]
    total_tokens = batch.completion_mask.sum().clamp(min=1).item()
    chunks = [list(range(i, min(i + micro_batch, N))) for i in range(0, N, micro_batch)]
    ctx = (lambda: torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype)) if autocast_dtype \
        else (lambda: torch.autocast(device_type="cpu", enabled=False))

    old, ref, ent_sum = [], [], 0.0
    model.eval()
    with torch.no_grad(), ctx():
        for idx in chunks:
            mb = batch.rows(idx).to(device)
            lp, ent = completion_logprobs(model, mb, with_entropy=True)
            old.append(lp)
            ent_sum += (ent * mb.completion_mask).sum().item()
            if ref_model is not None and beta > 0:
                ref.append(completion_logprobs(ref_model, mb)[0])
    model.train()

    agg = {"kl_sum": 0.0, "clip_sum": 0.0, "ratio_max": 0.0, "tokens": 0.0, "loss": 0.0, "grad_norm": 0.0}
    for _ in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for k, idx in enumerate(chunks):
            mb = batch.rows(idx).to(device)
            with ctx():
                lp, _ = completion_logprobs(model, mb)
                loss, st = grpo_loss(lp, old[k], ref[k] if ref else None, advantages[idx].to(device),
                                     mb.completion_mask, total_tokens, clip_eps, beta)
            loss.backward()
            agg["loss"] += loss.item()
            for key in ("kl_sum", "clip_sum", "tokens"):
                agg[key] += st[key]
            agg["ratio_max"] = max(agg["ratio_max"], st["ratio_max"])
        agg["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        optimizer.step()

    t = max(agg["tokens"], 1)
    return {"loss": agg["loss"] / ppo_epochs, "kl": agg["kl_sum"] / t, "clip_frac": agg["clip_sum"] / t,
            "ratio_max": agg["ratio_max"], "grad_norm": agg["grad_norm"], "entropy": ent_sum / total_tokens,
            "completion_tokens": total_tokens}
