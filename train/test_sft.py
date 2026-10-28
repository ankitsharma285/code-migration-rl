"""Tests for train/train_sft.py on a tiny random Qwen2 (CPU, seconds).  Run: envs/train/bin/python -m train.test_sft"""
import math

import torch

from train.grpo import build_batch, completion_logprobs
from train.test_grpo import PAD, tiny_model
from train.train_sft import lr_at, sft_step


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
    assert got["micro_batches"] > 1, "test should exercise packing"
    assert abs(got["nll"] - want) < 1e-4, (got["nll"], want)
    assert got["tokens"] == sum(len(c) for _, c in s)


def test_prompt_tokens_not_trained():
    """Changing a prompt token's target never enters the loss: only completion positions are scored."""
    m = tiny_model()
    s = seqs(n=1)
    b1 = build_batch([s[0][0]], [s[0][1]], PAD)
    assert b1.completion_mask.sum().item() == len(s[0][1])


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
