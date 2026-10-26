"""Multi-turn rollouts for training, in TOKEN space (v2, automatic execution).

Same episode rules as env/episode.py (which the harness uses for evaluation through an API):
  turn 1: the prompt (with feedback on the legacy code when feedback_prompt is on)
  turn k: the policy's reply -> run on the VISIBLE inputs -> feedback appended as a user turn
  stop:   visible examples pass | max_turns | two code-less replies in a row | context budget
  reward: the last attempt scored on the HIDDEN inputs

Why token space: GRPO needs the log-probs of exactly the tokens the policy sampled, in exactly the
context it sampled them in. Re-applying the chat template to the decoded text can re-tokenize a reply
differently, so each episode is kept as one growing token list:
  [turn-1 prompt] [reply 1] [feedback 1, as chat-template tokens] [reply 2] ...
with a mask that is 1 on reply tokens (trained on) and 0 on feedback tokens (conditioned on only).

Every turn is batched across all live episodes: one vLLM call per turn, sandbox runs in a thread pool.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from env.episode import FOLLOWUP, NO_CODE
from env.feedback import feedback
from env.reward import extract_code, score


class TurnTemplate:
    """The chat-template tokens that go between a reply and the next reply, derived from the tokenizer's own
    template with sentinel strings (so it works for any chat format, not just Qwen's)."""
    A, U1, U2 = "§A§", "§U1§", "§U2§"

    def __init__(self, tok):
        self.tok = tok
        text = tok.apply_chat_template([{"role": "user", "content": self.U1},
                                        {"role": "assistant", "content": self.A},
                                        {"role": "user", "content": self.U2}],
                                       add_generation_prompt=True, tokenize=False)
        after = text.split(self.A, 1)[1]                  # close-of-assistant + user turn + generation prompt
        self.prefix, self.suffix = after.split(self.U2, 1)
        eos = getattr(tok, "eos_token", None)
        self.eos_text = eos if eos and self.prefix.startswith(eos) else None

    def between(self, feedback_text, reply_ended_with_eos):
        """Token ids to append after a reply so the next turn starts exactly as the chat template would."""
        prefix = self.prefix
        if reply_ended_with_eos and self.eos_text:        # the sampled EOS already closed the assistant turn
            prefix = prefix[len(self.eos_text):]
        return list(self.tok.encode(prefix + FOLLOWUP.format(feedback=feedback_text) + self.suffix,
                                    add_special_tokens=False))


@dataclass
class Episode:
    item: dict
    prompt_ids: list
    cont_ids: list = field(default_factory=list)
    cont_mask: list = field(default_factory=list)
    texts: list = field(default_factory=list)          # each reply, decoded
    feedbacks: list = field(default_factory=list)      # environment reply after each turn (visible inputs)
    visible_ok: list = field(default_factory=list)
    hidden: list = field(default_factory=list)         # per-turn Reward on hidden inputs (only if score_turns)
    finished: list = field(default_factory=list)       # reply ended with EOS (False = cut at max_new_tokens)
    stop: str = ""
    final: object = None                               # Reward of the last attempt (hidden inputs)

    @property
    def turns(self):
        return len(self.texts)

    @property
    def reward(self):
        return self.final.reward if self.final is not None else 0.0


def rollout(gen, tok, tmpl, items, group_size, max_turns=3, temperature=1.0, top_p=1.0, seed=0,
            max_model_len=8192, max_new_tokens=1024, workers=16, score_turns=False, prompt_ids=None, timing=None):
    """items: prepared tasks (train/data.py). Returns len(items) * group_size Episodes, groups contiguous.
    timing (optional dict): filled with generate_s, feedback_s, score_s (wall time per phase)."""
    import time
    timing = timing if timing is not None else {}
    for key in ("generate_s", "feedback_s", "score_s"):
        timing.setdefault(key, 0.0)
    pids = prompt_ids or {}
    eps = []
    for it in items:
        p = pids.get(it["task"]["id"]) or tok_chat(tok, it["messages"])
        eps += [Episode(it, p) for _ in range(group_size)]
    pool = ThreadPoolExecutor(workers)
    try:
        for k in range(1, max_turns + 1):
            live = [e for e in eps if not e.stop]
            if not live:
                break
            fits = []
            for e in live:
                if len(e.prompt_ids) + len(e.cont_ids) + max_new_tokens > max_model_len:
                    e.stop = "context"
                else:
                    fits.append(e)
            if not fits:
                break
            if k == 1:      # every episode of a group shares the prompt: one request with n = group_size
                firsts = fits[::group_size] if len(fits) == len(eps) else None
            t0 = time.monotonic()
            if k == 1 and firsts is not None:
                outs = gen.generate([e.prompt_ids for e in firsts], n=group_size, temperature=temperature,
                                    top_p=top_p, seed=seed)
                samples = [s for group in outs for s in group]
            else:
                outs = gen.generate([e.prompt_ids + e.cont_ids for e in fits], n=1, temperature=temperature,
                                    top_p=top_p, seeds=[seed * 1000 + 7919 * k + i for i in range(len(fits))])
                samples = [g[0] for g in outs]
            timing["generate_s"] += time.monotonic() - t0
            t0 = time.monotonic()

            def env_reply(pair):
                e, (ids, text, done) = pair
                code = extract_code(text)
                if code is None:
                    return NO_CODE, False
                return feedback(e.item["task"], code, e.item["visible_inputs"], e.item["visible_refs"])
            fbs = list(pool.map(env_reply, zip(fits, samples)))
            timing["feedback_s"] += time.monotonic() - t0

            for e, (ids, text, done), (fb, ok) in zip(fits, samples, fbs):
                e.cont_ids += ids
                e.cont_mask += [1] * len(ids)
                e.texts.append(text)
                e.feedbacks.append(fb)
                e.visible_ok.append(ok)
                e.finished.append(done)
                if ok:
                    e.stop = "visible_pass"
                elif fb == NO_CODE and len(e.feedbacks) >= 2 and e.feedbacks[-2] == NO_CODE:
                    e.stop = "no_code"
                elif k == max_turns:
                    e.stop = "max_turns"
                else:
                    # did the sampled tokens already close the assistant turn? (vLLM includes the EOS id)
                    closed = bool(ids) and ids[-1] == getattr(tok, "eos_token_id", None)
                    tail = tmpl.between(fb, closed)
                    e.cont_ids += tail
                    e.cont_mask += [0] * len(tail)
        for e in eps:
            e.stop = e.stop or "max_turns"

        # rewards: the last attempt on the hidden inputs (every attempt when score_turns, for evaluation)
        t0 = time.monotonic()
        jobs = [(e, t) for e in eps for t in (range(e.turns) if score_turns else [e.turns - 1]) if e.turns]
        rs = list(pool.map(lambda j: score(j[0].item["task"], j[0].texts[j[1]], j[0].item["hidden_inputs"],
                                           j[0].item["hidden_refs"]), jobs))
        for (e, t), r in zip(jobs, rs):
            if score_turns:
                e.hidden.append(r)
            if t == e.turns - 1:
                e.final = r
        timing["score_s"] += time.monotonic() - t0
    finally:
        pool.shutdown()
    return eps


def template_consistent(tok, ep):
    """Does the episode's token list decode to what the chat template renders for the same conversation?
    Logged once per run: a mismatch means turn boundaries are tokenized differently from what the model saw
    in training (fix TurnTemplate before trusting a run). None when it cannot be checked."""
    if ep.turns < 2 or not all(ep.finished):
        return None
    msgs = list(ep.item["messages"])
    for k, text in enumerate(ep.texts):
        msgs.append({"role": "assistant", "content": text})
        if k < ep.turns - 1:
            msgs.append({"role": "user", "content": FOLLOWUP.format(feedback=ep.feedbacks[k])})
    try:
        want = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        got = tok.decode(ep.prompt_ids + ep.cont_ids, skip_special_tokens=False)
    except Exception:
        return None
    return got.strip() == want.strip()


def tok_chat(tok, messages):
    from train.data import chat_ids
    return chat_ids(tok, messages)
