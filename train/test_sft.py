"""Tests for train/train_sft.py on a tiny random Qwen2 (CPU, seconds).  Run: envs/train/bin/python -m train.test_sft"""
import math

import torch

from train.grpo import build_batch, completion_logprobs
from train.test_grpo import PAD, tiny_model
from train.train_sft import lr_at, right_padded, sequence_logprobs, sft_step


def seqs(seed=0, n=6):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        p = torch.randint(1, 64, (int(torch.randint(3, 9, (1,), generator=g)),), generator=g).tolist()
        c = torch.randint(1, 64, (int(torch.randint(2, 12, (1,), generator=g)),), generator=g).tolist()
        out.append((p, c))
    return out


def test_loss_is_token_mean_over_completions():
    m = tiny_model()
    s = seqs()
    b = build_batch([p for p, _ in s], [c for _, c in s], PAD)
    with torch.no_grad():
        lp, _ = completion_logprobs(m, b)
    want = -(lp * b.completion_mask).sum().item() / b.completion_mask.sum().item()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    got = sft_step(m, opt, s, PAD, max_tokens=24, max_rows=2, max_grad_norm=1e9, device="cpu")
    assert got["skipped"] == 0
    assert got["micro_batches"] > 1, "test should exercise packing"
    assert abs(got["nll"] - want) < 1e-4, (got["nll"], want)
    assert got["tokens"] == sum(len(c) for _, c in s)


def test_right_padding_matches_left_padded_grpo_logprobs():
    """The SFT path (right padding) scores each completion token exactly as GRPO's left-padded path does."""
    m = tiny_model()
    s = seqs()
    b = build_batch([p for p, _ in s], [c for _, c in s], PAD)
    ids, attn, mask = right_padded(s, PAD)
    with torch.no_grad():
        left, _ = completion_logprobs(m, b)
        for chunk in (None, 3):
            right = sequence_logprobs(m, ids, attn, chunk)
            for i, (p, c) in enumerate(s):
                a = left[i, :len(c)]
                r = right[i, len(p) - 1:len(p) + len(c) - 1]
                assert torch.allclose(a, r, atol=1e-5), (i, a, r)
                assert mask[i].sum().item() == len(c)
                assert mask[i, len(p) - 1].item() == 1 and (len(p) < 2 or mask[i, len(p) - 2].item() == 0)


def test_nonfinite_step_is_skipped():
    m = tiny_model()
    with torch.no_grad():
        m.lm_head.weight[3, 0] = float("nan")
    before = {k: v.clone() for k, v in m.state_dict().items()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    st = sft_step(m, opt, seqs(), PAD, 4096, 8, 1.0, "cpu")
    assert st["skipped"] == 1
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) or torch.isnan(before[k]).any() for k in before)
    assert torch.equal(before["model.embed_tokens.weight"], after["model.embed_tokens.weight"])
    assert st["fp32_retry"] == 1


def test_bf16_nan_gradient_recomputed_in_fp32():
    """A NaN that appears only in the bf16-autocast backward: the step is redone in fp32 and applied."""
    m = tiny_model()
    seen = {}
    def note(module, args, out):                       # was this forward run under autocast?
        seen["bf16"] = torch.is_autocast_enabled("cpu")
    def poison(module, grad_in, grad_out):            # then poison its backward
        if seen.get("bf16"):
            return tuple(torch.full_like(g, float("nan")) if g is not None else None for g in grad_in)
    mlp = m.model.layers[0].mlp
    mlp.register_forward_hook(note)
    mlp.register_full_backward_hook(poison)
    ref = tiny_model()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    opt_ref = torch.optim.SGD(ref.parameters(), lr=1e-2)
    st = sft_step(m, opt, seqs(), PAD, 4096, 8, 1.0, "cpu", autocast_dtype=torch.bfloat16)
    st_ref = sft_step(ref, opt_ref, seqs(), PAD, 4096, 8, 1.0, "cpu")      # plain fp32 step
    assert st["fp32_retry"] == 1 and st["skipped"] == 0, st
    assert st["nan_at"] is not None and st["nan_at"]["micro"] == 0
    assert all(torch.isfinite(p).all() for p in m.parameters())
    for (n, a), b in zip(m.named_parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=1e-6), n


def test_nll_decreases_when_fitting_one_batch():
    m = tiny_model()
    s = seqs()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = sft_step(m, opt, s, PAD, 4096, 8, 1.0, "cpu")["nll"]
    for _ in range(30):
        last = sft_step(m, opt, s, PAD, 4096, 8, 1.0, "cpu")["nll"]
    assert last < 0.5 * first, (first, last)


def test_lr_schedule():
    assert abs(lr_at(0, 100, 10, 1.0) - 0.1) < 1e-9
    assert abs(lr_at(9, 100, 10, 1.0) - 1.0) < 1e-9
    assert abs(lr_at(10, 100, 10, 1.0) - 1.0) < 1e-9
    assert abs(lr_at(55, 100, 10, 1.0) - 0.5) < 1e-9
    assert lr_at(100, 100, 10, 1.0) < 1e-9
    assert all(lr_at(i, 100, 10, 1.0) >= lr_at(i + 1, 100, 10, 1.0) for i in range(10, 100))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
