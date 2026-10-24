"""Day-6 gate: does the trainer -> vLLM weight sync actually work on this vLLM version?

On one GPU (default: Qwen2.5-0.5B-Instruct, small and fast):
  1. load vLLM and the trainer with the same weights; their log-probs on fixed completions should agree
  2. perturb the trainer's weights WITHOUT syncing: the log-probs must now disagree (proves the check
     can detect a stale vLLM)
  3. sync: the log-probs must agree again (proves the sync took effect)
A sync that silently does nothing fails step 3, so this cannot pass by accident.
Pass: steps 1 and 3 within --tol, and the stale gap (step 2) at least 5x the larger of them.
After sync the gap is slightly above step 1's: perturbed fp32 weights round when cast to bf16 for vLLM.

Run on a GPU node:  envs/train/bin/python -m train.check_sync [--model Qwen/Qwen2.5-0.5B-Instruct]
"""
import argparse

import torch

from train.grpo import build_batch, completion_logprobs


def trainer_logprobs(model, prompts, comps, pad, device):
    b = build_batch(prompts, comps, pad).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lp, _ = completion_logprobs(model, b)
    return [[lp[i, j].item() for j in range(len(c))] for i, c in enumerate(comps)]


def gap(a, b):
    d = [abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    return sum(d) / len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tol", type=float, default=0.05, help="max mean |logprob gap| (nats) that counts as agreement")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train.data import chat_ids
    from train.generation import VLLMGenerator

    gen = VLLMGenerator(args.model, max_model_len=2048, max_new_tokens=32, gpu_memory_utilization=0.3)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda().eval()

    msgs = [[{"role": "user", "content": "Write a pandas function that sums a column."}],
            [{"role": "user", "content": "Replace np.float with the builtin in: x = np.float(3)"}]]
    prompts = [chat_ids(tok, m) for m in msgs]
    comps = [s[0][0] for s in gen.generate(prompts, n=1, temperature=0.0)]

    v0 = gen.logprobs_of(prompts, comps)
    g_initial = gap(trainer_logprobs(policy, prompts, comps, pad, "cuda"), v0)

    with torch.no_grad():                                   # a large, deliberate change
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * p.abs().mean() * 0.2)
    t_new = trainer_logprobs(policy, prompts, comps, pad, "cuda")
    g_stale = gap(t_new, gen.logprobs_of(prompts, comps))

    gen.sync_weights(policy)
    g_synced = gap(t_new, gen.logprobs_of(prompts, comps))

    print(f"mean |logprob gap| (nats): initial {g_initial:.4f}   after perturbation, no sync {g_stale:.4f}   "
          f"after sync {g_synced:.4f}")
    # relative test: after sync the gap must be back at noise level, and far below the stale gap
    noise = max(g_initial, g_synced)
    ok = g_initial <= args.tol and g_synced <= args.tol and g_stale > 5 * noise
    print("SYNC CHECK: PASS" if ok else "SYNC CHECK: FAIL")
    if not ok:
        if g_initial > args.tol:
            print("  trainer and vLLM disagree even before any change: check dtype/positions, or raise --tol")
        elif g_stale <= 5 * noise:
            print("  the perturbation did not change log-probs enough for a meaningful test")
        else:
            print("  vLLM still shows the OLD weights after sync: sync_weights() did not take effect")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
