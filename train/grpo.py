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


def build_batch(prompt_ids, completion_ids, pad_id, completion_masks=None):
    """prompt_ids, completion_ids: lists of token-id lists (completion i belongs to prompt i).
    completion_masks (optional, multi-turn): per-token 0/1 lists aligned with completion_ids; 1 = a token
    the policy generated (trained on), 0 = environment text inside the episode (feedback turns: the
    policy is conditioned on it but never trained to produce it)."""
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
            cmask[i, :len(c)] = 1.0 if completion_masks is None else torch.tensor(
                completion_masks[i], dtype=torch.float)
    pos = (attn.cumsum(dim=1) - 1).clamp(min=0)
    return Batch(ids, attn, pos, cmask, P)


# ---------------------------------------------------------------- log-probs

def _token_logp(lm_head, h, targets, with_entropy):
    logits = lm_head(h).float()
    logZ = torch.logsumexp(logits, dim=-1)
    logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logZ
    if not with_entropy:
        return logp, None
    probs = torch.softmax(logits, dim=-1)
    return logp, (logZ - (probs * logits).sum(-1)).detach()


def completion_logprobs(model, batch, with_entropy=False, chunk=None):
    """Log-prob of each completion token under `model`: [N, C]. Optionally the per-token entropy.
    Only the completion span's logits are materialized for the log-softmax (vocab is ~150k).
    chunk=K (multi-turn episodes are long): run the LM head over K positions at a time, recomputing each
    chunk's logits in backward (activation checkpointing), so full-vocab logits never exist for the whole
    sequence at once. Same numbers as the unchunked path."""
    P = batch.prompt_len
    if chunk and hasattr(model, "model") and hasattr(model, "lm_head"):
        h = model.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                        position_ids=batch.position_ids, use_cache=False).last_hidden_state[:, P - 1:-1, :]
        targets = batch.input_ids[:, P:]
        lps, ents = [], []
        for a in range(0, h.shape[1], chunk):
            hc, tc = h[:, a:a + chunk], targets[:, a:a + chunk]
            if torch.is_grad_enabled():
                lp, ent = torch.utils.checkpoint.checkpoint(_token_logp, model.lm_head, hc, tc, with_entropy,
                                                            use_reentrant=False)
            else:
                lp, ent = _token_logp(model.lm_head, hc, tc, with_entropy)
            lps.append(lp)
            ents.append(ent)
        return torch.cat(lps, 1), (torch.cat(ents, 1) if with_entropy else None)
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                position_ids=batch.position_ids, use_cache=False)
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
    chunks = [list(range(i, min(i + micro_batch, N))) for i in range(0, N, micro_batch)]
    micro = [(batch.rows(idx), advantages[idx]) for idx in chunks]
    return _update(model, optimizer, micro, ref_model, beta, clip_eps, ppo_epochs, max_grad_norm, device,
                   autocast_dtype, None)


def pack(lengths, max_tokens, max_rows):
    """Group sequence indices into micro-batches whose padded size (rows x longest row) stays within
    max_tokens. Sorting by length keeps padding small. A sequence longer than max_tokens rides alone."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * lengths[i] > max_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def grpo_step_packed(model, optimizer, sequences, advantages, pad_id, ref_model=None, beta=0.0, clip_eps=0.2,
                     max_tokens=16384, max_rows=8, ppo_epochs=1, max_grad_norm=1.0, device="cpu",
                     autocast_dtype=None, logit_chunk=None):
    """Multi-turn version of grpo_step. sequences: list of (prompt_ids, continuation_ids, continuation_mask):
    the continuation holds every turn's completion AND the feedback in between, and the mask is 1 only
    on the policy's own tokens. Micro-batches are packed by a token budget (episodes vary from a few
    hundred to ~8k tokens) and padded per micro-batch, not to the longest episode in the step."""
    lengths = [len(p) + len(c) for p, c, _ in sequences]
    micro = []
    for idx in pack(lengths, max_tokens, max_rows):
        b = build_batch([sequences[i][0] for i in idx], [sequences[i][1] for i in idx], pad_id,
                        [sequences[i][2] for i in idx])
        micro.append((b, advantages[idx]))
    return _update(model, optimizer, micro, ref_model, beta, clip_eps, ppo_epochs, max_grad_norm, device,
                   autocast_dtype, logit_chunk)


def _update(model, optimizer, micro, ref_model, beta, clip_eps, ppo_epochs, max_grad_norm, device,
            autocast_dtype, logit_chunk):
    total_tokens = sum(b.completion_mask.sum().item() for b, _ in micro) or 1.0
    ctx = (lambda: torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype)) if autocast_dtype \
        else (lambda: torch.autocast(device_type="cpu", enabled=False))

    old, ref, ent_sum = [], [], 0.0
    model.eval()
    with torch.no_grad(), ctx():
        for b, _ in micro:
            mb = b.to(device)
            lp, ent = completion_logprobs(model, mb, with_entropy=True, chunk=logit_chunk)
            old.append(lp)
            ent_sum += (ent * mb.completion_mask).sum().item()
            if ref_model is not None and beta > 0:
                ref.append(completion_logprobs(ref_model, mb, chunk=logit_chunk)[0])
    model.train()

    agg = {"kl_sum": 0.0, "clip_sum": 0.0, "ratio_max": 0.0, "tokens": 0.0, "loss": 0.0, "grad_norm": 0.0}
    for _ in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for k, (b, adv) in enumerate(micro):
            mb = b.to(device)
            with ctx():
                lp, _ = completion_logprobs(model, mb, chunk=logit_chunk)
                loss, st = grpo_loss(lp, old[k], ref[k] if ref else None, adv.to(device),
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
            "completion_tokens": total_tokens, "micro_batches": len(micro)}
