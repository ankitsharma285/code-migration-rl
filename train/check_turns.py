"""Check, with the REAL tokenizer, that multi-turn episodes are tokenized exactly as the chat template would
render the same conversation (CPU, seconds; needs the model's tokenizer in the HF cache).

A mismatch would mean the policy is trained on turn boundaries it never sees at inference time.

  envs/train/bin/python -m train.check_turns [--model Qwen/Qwen2.5-Coder-1.5B-Instruct]
"""
import argparse

from transformers import AutoTokenizer

from train.data import chat_ids
from train.rollout import Episode, TurnTemplate, template_consistent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    t = TurnTemplate(tok)
    print(f"between-turn text: {t.prefix!r} + <feedback message> + {t.suffix!r}; eos {tok.eos_token!r} "
          f"(dropped from the prefix after a sampled EOS: {t.eos_text is not None})")
    msgs = [{"role": "system", "content": "You migrate code."}, {"role": "user", "content": "Migrate f."}]
    replies = ["```python\ndef f(df):\n    return df.items()\n```", "```python\ndef f(df):\n    return df.map(str)\n```",
               "Here:\n```python\ndef f(df):\n    return 1\n```"]
    fbs = ["Example 1: raised AttributeError: 'DataFrame' object has no attribute 'iteritems'",
           "Example 1: wrong output.\n  expected: {\"t\": \"int\", \"value\": 1}\n  got:      {}"]
    ok = True
    for eos in (True, False):
        ids, mask = [], []
        for k, r in enumerate(replies):
            rid = tok.encode(r, add_special_tokens=False) + ([tok.eos_token_id] if eos else [])
            ids += rid
            mask += [1] * len(rid)
            if k < len(replies) - 1:
                mid = t.between(fbs[k], eos)
                ids += mid
                mask += [0] * len(mid)
        e = Episode({"messages": msgs}, chat_ids(tok, msgs), ids, mask, texts=replies, feedbacks=fbs + ["-"],
                    finished=[True] * 3)
        res = template_consistent(tok, e) if eos else None
        if eos:
            print(f"replies ending in EOS: token list matches the chat template: {res}")
            ok &= bool(res)
        # the feedback text must survive the round trip, and the mask must cover replies only
        text = tok.decode([i for i, m in zip(ids, mask) if m == 0])
        ok &= all(f.split("\n")[0] in text for f in fbs)
        ok &= sum(mask) == sum(len(tok.encode(r, add_special_tokens=False)) + (1 if eos else 0) for r in replies)
    print("TURN CHECK: PASS" if ok else "TURN CHECK: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
