"""Unit tests for train/grpo.py on a tiny randomly initialized Qwen2 model (CPU, seconds).

Run:  envs/train/bin/python -m train.test_grpo        (no pytest needed)

The first three are the plan's required tests; the rest catch the usual silent bugs in RL code.
"""
import copy
import math

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from train.grpo import (build_batch, completion_logprobs, group_advantages, grpo_loss, grpo_step,
                        grpo_step_packed, mixed_group_fraction, pack)

PAD = 0


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).float()


def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


# ---------------------------------------------------------------- required by the plan

def test_advantage_zero_for_uniform_groups():
    r = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.])
    a = group_advantages(r, group_size=4)
    assert torch.all(a[:8] == 0), "all-same groups must give zero advantage"
    assert a[8] > 0 and torch.all(a[9:12] < 0)
    assert abs(a[8:12].sum().item()) < 1e-5, "advantages are centered within a group"
    assert abs(mixed_group_fraction(r, 4) - 1 / 3) < 1e-6   # float32


def test_ratio_is_one_on_first_pass_and_gradient_is_policy_gradient():
    model = tiny_model()
    b = build_batch([[5, 6, 7], [8, 9]], [[10, 11, 12], [13]], PAD)
    adv = torch.tensor([1.0, -0.5])
    logp, _ = completion_logprobs(model, b)
    old = logp.detach()
    T = b.completion_mask.sum()
    loss, st = grpo_loss(logp, old, None, adv, b.completion_mask, T, clip_eps=0.2, beta=0.0)
    assert abs(st["ratio_max"] - 1.0) < 1e-6
    model.zero_grad(); loss.backward(); g1 = grads(model).clone()
    # plain policy gradient: -sum(A * logp) / T has the same gradient when ratio == 1
    logp2, _ = completion_logprobs(model, b)
    pg = -(adv.unsqueeze(1) * logp2 * b.completion_mask).sum() / T
    model.zero_grad(); pg.backward(); g2 = grads(model)
    assert torch.allclose(g1, g2, atol=1e-6), "GRPO gradient at ratio=1 must equal the policy gradient"


def test_one_step_moves_logprobs_toward_advantage():
    model = tiny_model()
    b = build_batch([[5, 6, 7], [5, 6, 7]], [[20, 21, 22], [30, 31, 32]], PAD)
    adv = torch.tensor([1.0, -1.0])                      # first completion good, second bad
    before = completion_logprobs(model, b)[0].sum(1).detach()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    grpo_step(model, opt, b, adv, micro_batch=2, max_grad_norm=1e9)
    after = completion_logprobs(model, b)[0].sum(1).detach()
    assert after[0] > before[0], "log-prob of the positive-advantage completion must rise"
    assert after[1] < before[1], "log-prob of the negative-advantage completion must fall"


# ---------------------------------------------------------------- silent-bug guards

def test_left_padding_does_not_change_logprobs():
    model = tiny_model()
    alone = build_batch([[5, 6, 7]], [[10, 11, 12]], PAD)
    padded = build_batch([[5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]], [[10, 11, 12], [9]], PAD)
    a = completion_logprobs(model, alone)[0][0]
    p = completion_logprobs(model, padded)[0][0, :3]
    assert torch.allclose(a, p, atol=1e-5), "a longer prompt elsewhere in the batch must not shift positions"


def test_padding_tokens_do_not_affect_loss():
    model = tiny_model()
    b = build_batch([[5, 6], [7, 8]], [[10, 11, 12, 13], [14]], PAD)
    adv = torch.tensor([0.7, -0.7])
    T = b.completion_mask.sum()
    lp = completion_logprobs(model, b)[0]
    l1, _ = grpo_loss(lp, lp.detach(), None, adv, b.completion_mask, T)
    b.input_ids[1, b.prompt_len + 1:] = 42               # garbage in row 1's completion padding
    lp2 = completion_logprobs(model, b)[0]
    l2, _ = grpo_loss(lp2, lp2.detach(), None, adv, b.completion_mask, T)
    assert abs(l1.item() - l2.item()) < 1e-6, "completion padding must be masked out"


def test_microbatch_accumulation_matches_full_batch():
    prompts = [[5, 6, 7], [8, 9], [3, 4, 5, 6], [7]]
    comps = [[10, 11], [12, 13, 14], [15], [16, 17, 18, 19]]
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    m1, m2 = tiny_model(), tiny_model()
    o1, o2 = torch.optim.SGD(m1.parameters(), lr=0.1), torch.optim.SGD(m2.parameters(), lr=0.1)
    b = build_batch(prompts, comps, PAD)
    grpo_step(m1, o1, b, adv, micro_batch=4, max_grad_norm=1e9)
    grpo_step(m2, o2, b, adv, micro_batch=1, max_grad_norm=1e9)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "micro-batching must not change the update"


def test_kl_zero_against_itself_and_positive_otherwise():
    model, other = tiny_model(0), tiny_model(1)
    b = build_batch([[5, 6, 7]], [[10, 11, 12]], PAD)
    lp = completion_logprobs(model, b)[0]
    T = b.completion_mask.sum()
    _, same = grpo_loss(lp, lp.detach(), lp.detach(), torch.tensor([0.0]), b.completion_mask, T, beta=0.1)
    _, diff = grpo_loss(lp, lp.detach(), completion_logprobs(other, b)[0].detach(), torch.tensor([0.0]),
                        b.completion_mask, T, beta=0.1)
    assert abs(same["kl_sum"]) < 1e-6 and diff["kl_sum"] > 0


def test_clipping_zeroes_gradient_past_the_trust_region():
    logp = torch.tensor([[math.log(1.5)]], requires_grad=True)     # ratio 1.5 vs old logp 0
    old = torch.tensor([[0.0]])
    mask = torch.ones(1, 1)
    loss, st = grpo_loss(logp, old, None, torch.tensor([1.0]), mask, 1, clip_eps=0.2)
    loss.backward()
    assert logp.grad.abs().item() == 0.0 and st["clip_sum"] == 1.0, "ratio > 1+eps with A > 0 is clipped"
    logp2 = torch.tensor([[math.log(1.1)]], requires_grad=True)    # inside the region: gradient flows
    grpo_loss(logp2, old, None, torch.tensor([1.0]), mask, 1, clip_eps=0.2)[0].backward()
    assert logp2.grad.abs().item() > 0


def test_completion_logprobs_match_a_direct_computation():
    model = tiny_model()
    b = build_batch([[5, 6, 7]], [[10, 11, 12]], PAD)
    lp = completion_logprobs(model, b)[0][0]
    full = torch.log_softmax(model(input_ids=torch.tensor([[5, 6, 7, 10, 11, 12]])).logits[0], -1)
    direct = torch.stack([full[2, 10], full[3, 11], full[4, 12]])
    assert torch.allclose(lp, direct, atol=1e-5)


# ---------------------------------------------------------------- multi-turn (v2)

def test_chunked_logprobs_match_unchunked_values_and_gradients():
    b = build_batch([[5, 6, 7], [8, 9]], [[10, 11, 12, 13, 14], [15, 16]], PAD)
    m1, m2 = tiny_model(), tiny_model()
    a, _ = completion_logprobs(m1, b)
    c, _ = completion_logprobs(m2, b, chunk=2)
    assert torch.allclose(a, c, atol=1e-5), "chunked log-probs must equal the full computation"
    (a * b.completion_mask).sum().backward()
    (c * b.completion_mask).sum().backward()
    assert torch.allclose(grads(m1), grads(m2), atol=1e-5), "chunked backward must give the same gradient"


def test_masked_feedback_tokens_carry_no_loss():
    """Environment tokens (mask 0) inside an episode: the loss must equal a sum over the policy's tokens only."""
    model = tiny_model()
    prompt, cont, mask = [5, 6, 7], [10, 11, 20, 21, 22, 12, 13], [1, 1, 0, 0, 0, 1, 1]
    b = build_batch([prompt], [cont], PAD, [mask])
    lp = completion_logprobs(model, b)[0]
    loss, st = grpo_loss(lp, lp.detach(), None, torch.tensor([1.0]), b.completion_mask, 4)
    manual = -(lp[0, [0, 1, 5, 6]]).exp().div(lp[0, [0, 1, 5, 6]].detach().exp()).sum() / 4
    assert st["tokens"] == 4 and torch.allclose(loss, manual, atol=1e-6), "feedback tokens must be excluded"


def test_packed_step_matches_plain_step():
    prompts = [[5, 6, 7], [8, 9], [3, 4, 5, 6], [7]]
    comps = [[10, 11], [12, 13, 14], [15], [16, 17, 18, 19]]
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    m1, m2 = tiny_model(), tiny_model()
    o1, o2 = torch.optim.SGD(m1.parameters(), lr=0.1), torch.optim.SGD(m2.parameters(), lr=0.1)
    grpo_step(m1, o1, build_batch(prompts, comps, PAD), adv, micro_batch=4, max_grad_norm=1e9)
    seqs = [(p, c, [1] * len(c)) for p, c in zip(prompts, comps)]
    st = grpo_step_packed(m2, o2, seqs, adv, PAD, max_tokens=10, max_rows=8, max_grad_norm=1e9, logit_chunk=2)
    assert st["micro_batches"] > 1, "the token budget should split this batch"
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "packing + chunking must not change the update"


def test_pack_respects_token_budget():
    lengths = [100, 900, 300, 50, 800, 400]
    groups = pack(lengths, max_tokens=1000, max_rows=8)
    assert sorted(i for g in groups for i in g) == list(range(6))
    assert all(len(g) * max(lengths[i] for i in g) <= 1000 for g in groups)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
