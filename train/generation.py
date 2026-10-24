"""Rollout generators. Each returns, per prompt, `n` samples as (completion_token_ids, text, finished).

  VLLMGenerator  vLLM in the SAME process and on the SAME GPU as the trainer (one-GPU setup). After
                 each optimizer step, sync_weights() copies the trainer's weights into vLLM.
  HFGenerator    model.generate(); slow, for CPU tests and debugging only.

The weight sync touches vLLM internals, which change between versions, so it lives in one function
and is verified by verify_sync(): after a sync, vLLM's log-probs for a fixed completion must match
the trainer's (the plan's day-6 gate).
"""
import os

import torch


class HFGenerator:
    def __init__(self, model, tokenizer, max_new_tokens=1024):
        self.model, self.tok, self.max_new_tokens = model, tokenizer, max_new_tokens

    @torch.no_grad()
    def generate(self, prompt_ids, n, temperature=1.0, top_p=1.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        out = []
        self.model.eval()
        for p in prompt_ids:
            ids = torch.tensor([p] * n)
            g = self.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=temperature > 0,
                                    temperature=max(temperature, 1e-5), top_p=top_p,
                                    max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.tok.eos_token_id)
            samples = []
            for row in g[:, len(p):].tolist():
                if self.tok.eos_token_id in row:
                    row = row[: row.index(self.tok.eos_token_id) + 1]
                    finished = True
                else:
                    finished = False
                row = [t for t in row if t != self.tok.pad_token_id or t == self.tok.eos_token_id]
                samples.append((row, self.tok.decode(row, skip_special_tokens=True), finished))
            out.append(samples)
        return out

    def sync_weights(self, model):
        pass                                     # generates with the training model itself


class VLLMGenerator:
    def __init__(self, model_path, max_model_len=4096, max_new_tokens=1024, gpu_memory_utilization=0.35,
                 seed=0, enforce_eager=False):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # engine in this process: needed
        from vllm import LLM                                           # to reach its model for weight sync
        self.llm = LLM(model=model_path, dtype="bfloat16", max_model_len=max_model_len, seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization, enforce_eager=enforce_eager,
                       enable_prefix_caching=True)
        self.max_new_tokens = max_new_tokens
        self.eos = self.llm.get_tokenizer().eos_token_id

    def generate(self, prompt_ids, n, temperature=1.0, top_p=1.0, seed=None):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=self.max_new_tokens, seed=seed)
        try:
            from vllm.inputs import TokensPrompt
            prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_ids]
        except ImportError:
            prompts = [{"prompt_token_ids": p} for p in prompt_ids]
        results = self.llm.generate(prompts, sp, use_tqdm=False)
        out = []
        for r in results:
            out.append([(list(o.token_ids), o.text, o.finish_reason == "stop") for o in r.outputs])
        return out

    def _vllm_model(self):
        """The nn.Module vLLM samples from. Attribute paths differ across vLLM versions; try known ones."""
        engine = self.llm.llm_engine
        candidates = [
            lambda: engine.model_executor.driver_worker.worker.model_runner.model,
            lambda: engine.model_executor.driver_worker.model_runner.model,
            lambda: engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
        ]
        for get in candidates:
            try:
                m = get()
                if hasattr(m, "load_weights"):
                    return m
            except AttributeError:
                continue
        raise RuntimeError("Could not reach vLLM's model for weight sync in this vLLM version. Make sure "
                           "VLLM_ENABLE_V1_MULTIPROCESSING=0 is set before vllm is imported, then inspect "
                           "self.llm.llm_engine to find the model_runner path and add it to _vllm_model().")

    @torch.no_grad()
    def sync_weights(self, model):
        """Copy the trainer's current weights (cast to bf16) into vLLM, one tensor at a time."""
        target = self._vllm_model()
        target.load_weights(((name, p.detach().to(torch.bfloat16)) for name, p in model.state_dict().items()))
        try:
            self.llm.reset_prefix_cache()       # cached KV was computed with the old weights
        except AttributeError:
            pass

    def logprobs_of(self, prompt_ids, completion_ids):
        """vLLM's log-probs of a fixed completion (prompt_logprobs over the joined sequence)."""
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        joined = [p + c for p, c in zip(prompt_ids, completion_ids)]
        try:
            from vllm.inputs import TokensPrompt
            reqs = [TokensPrompt(prompt_token_ids=j) for j in joined]
        except ImportError:
            reqs = [{"prompt_token_ids": j} for j in joined]
        res = self.llm.generate(reqs, sp, use_tqdm=False)
        out = []
        for r, p, c in zip(res, prompt_ids, completion_ids):
            plp = r.prompt_logprobs                      # list aligned with tokens; entry 0 is None
            out.append([plp[len(p) + i][c[i]].logprob for i in range(len(c))])
        return out


def verify_sync(generator, model, tokenizer, prompts, completions, device, tol=0.05):
    """Mean absolute difference between trainer and vLLM log-probs on fixed completions. bf16 kernels
    differ slightly, so a small gap is normal; a gap near or above 1 nat means the sync failed."""
    from train.grpo import build_batch, completion_logprobs
    generator.sync_weights(model)
    v = generator.logprobs_of(prompts, completions)
    b = build_batch(prompts, completions, tokenizer.pad_token_id).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        t, _ = completion_logprobs(model, b)
    diffs = [abs(t[i, j].item() - v[i][j]) for i in range(len(v)) for j in range(len(v[i]))]
    gap = sum(diffs) / len(diffs)
    return gap, gap <= tol
