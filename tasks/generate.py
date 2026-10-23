"""Generate candidate migration tasks with an LLM behind an OpenAI-compatible endpoint (e.g. vLLM).

Each request asks for ONE task built around 1-3 requested migration types, in a realistic domain,
in a tagged format (ENTRY / LEGACY / MODERN / GEN). Nothing here is trusted: tasks/filter_tasks.py
re-derives the task's real types from the code and runs every check before a task is kept.

Usage (from the repo root, with a server running):
  envs/legacy/bin/python -m tasks.generate --base-url http://localhost:8000/v1 --model gemma4 \
      --n 50 --split train --out tasks/generated/raw_pilot.jsonl
"""
import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.policies import OpenAIPolicy
from tasks.seed_tasks import SEED_TASKS
from tasks.verify_types import PROBES

HERE = Path(__file__).resolve().parent
CATS = json.loads((HERE / "categories.json").read_text())
TRAIN_BEHAVIOR = CATS["train"]["behavior"]
TRAIN_RENAME = CATS["train"]["rename"]
HELD_OUT = CATS["held_out"]
PROBE = {p["id"]: p for p in PROBES}

DOMAINS = [
    "retail sales", "warehouse inventory", "web analytics", "IoT sensor readings", "payroll",
    "clinical trial visits", "sports league statistics", "energy meter data", "airline bookings",
    "customer support tickets", "stock price history", "school grades", "ride-sharing trips",
    "restaurant orders", "manufacturing quality control", "real estate listings", "weather stations",
    "email campaign metrics", "library loans", "crop yields",
]

SYSTEM = ("You write realistic, small Python data-processing functions that use pandas and NumPy, "
          "together with their exact migrations to newer library versions. You follow the output "
          "format precisely.")

TEMPLATE = """Write ONE realistic {domain} function written for pandas 1.3 / NumPy 1.21 that must be migrated to pandas 2.2 / NumPy 2.x.

The legacy function MUST use these deprecated or removed APIs, in a natural way:
{type_block}

Requirements:
- Pure function: same arguments -> same result. No randomness, printing, files, network, or global state.
- `pd`, `np`, and `io` are pre-imported; do not import anything else.
- 5 to 25 lines. Realistic names. Return a DataFrame, Series, ndarray, scalar, list, or dict.
- The migrated function behaves IDENTICALLY on every input: same values, dtypes, index, and names,
  and raises no DeprecationWarning or FutureWarning on pandas 2.2 / NumPy 2.x.
- Do not use: DataFrame.mad, Series.dt.weekofyear / .dt.week, DataFrame.lookup, private attributes (names starting with _), warnings, try/except.
- Avoid building DataFrames by appending to an EMPTY DataFrame (it creates object dtype columns).

The input generator `gen(rng)` receives `rng = random.Random(seed)` and returns the keyword arguments as plain JSON data:
- a DataFrame is written as {{"__df__": {{"col": [values...], ...}}}}
- a Series is written as {{"__series__": {{"values": [...], "index": [...], "name": "..."}}}} (index optional; a list of lists makes a MultiIndex)
- use None for missing values (never float('nan'))
- 2 to 15 rows; only Python's `random` module (via rng) and built-ins
- MOST inputs must exercise the deprecated code path (e.g. always include at least one missing value if the code fills them).

Use exactly this format:

### ENTRY
<function name>

### LEGACY
```python
<legacy function>
```

### MODERN
```python
<migrated function, same name and arguments>
```

### GEN
```python
def gen(rng):
    ...
```

Here is a complete example of the format (different APIs and domain):

### ENTRY
{ex_entry}

### LEGACY
```python
{ex_legacy}
```

### MODERN
```python
{ex_modern}
```

### GEN
```python
{ex_gen}
```"""


def type_line(type_id):
    p = PROBE[type_id]
    legacy = p["legacy"].replace("result = ", "").strip()
    modern = p["modern"].replace("result = ", "").strip()
    return (f"- {type_id} ({p['history']})\n    old: {legacy}\n    new: {modern}")


def pick_types(rng, split):
    if split == "heldout":
        return [rng.choice(HELD_OUT)]
    k = rng.choice([1, 2, 2, 3])
    types = set()
    if rng.random() < 0.65:                      # target >= 60% of tasks with a behavior-change type
        types.add(rng.choice(TRAIN_BEHAVIOR))
    while len(types) < k:
        types.add(rng.choice(TRAIN_BEHAVIOR + TRAIN_RENAME))
    return sorted(types)


def build_messages(types, domain, example):
    user = TEMPLATE.format(domain=domain, type_block="\n".join(type_line(t) for t in types),
                           ex_entry=example["entry"], ex_legacy=example["legacy"].strip(),
                           ex_modern=example["modern"].strip(), ex_gen=example["gen"].strip())
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_SECTION = re.compile(r"###\s*(ENTRY|LEGACY|MODERN|GEN)\s*\n(.*?)(?=\n###\s*(?:ENTRY|LEGACY|MODERN|GEN)\b|\Z)", re.S)
_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def parse(text):
    """Returns (task_fields, error)."""
    sections = {name: body.strip() for name, body in _SECTION.findall(text or "")}
    missing = [s for s in ("ENTRY", "LEGACY", "MODERN", "GEN") if s not in sections]
    if missing:
        return None, f"missing sections: {missing}"
    out = {"entry": sections["ENTRY"].strip("` \n").split()[0] if sections["ENTRY"].strip("` \n") else ""}
    for name in ("LEGACY", "MODERN", "GEN"):
        m = _CODE.search(sections[name])
        if not m:
            return None, f"no code block in {name}"
        out[name.lower()] = m.group(1).rstrip() + "\n"
    if not re.fullmatch(r"[A-Za-z]\w*", out["entry"]):
        return None, f"bad entry name {out['entry']!r}"
    return out, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--seed", type=int, default=0, help="controls which types/domains are requested")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16, help="parallel requests to the server")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(f"{args.split}-{args.seed}")
    examples = [t for t in SEED_TASKS]                      # seed tasks use training types only
    requests = []
    for i in range(args.n):
        types = pick_types(rng, args.split)
        domain = rng.choice(DOMAINS)
        # format example: a seed task sharing no type with the request, so the answer isn't copied
        ex = rng.choice([e for e in examples if not set(e["types"]) & set(types)] or examples)
        requests.append({"request_id": f"{args.split}-{args.seed}-{i:05d}", "types": types,
                         "domain": domain, "example": ex["id"],
                         "messages": build_messages(types, domain, ex)})

    policy = OpenAIPolicy(args.model, args.base_url, args.api_key_env, args.temperature, args.max_tokens)

    def one(req):
        t0 = time.monotonic()
        try:
            text = policy.sample(req["messages"], 1, None)[0]
            err = ""
        except Exception as e:                            # keep going; record the failure
            text, err = "", f"request failed: {type(e).__name__}: {e}"[:300]
        task, perr = parse(text) if not err else (None, err)
        return {"request_id": req["request_id"], "split": args.split, "requested_types": req["types"],
                "domain": req["domain"], "model": args.model, "raw": text, "task": task,
                "parse_error": perr, "gen_s": round(time.monotonic() - t0, 2)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_ok = 0
    with ThreadPoolExecutor(args.workers) as pool, open(out, "w") as f:
        for rec in pool.map(one, requests):
            n_ok += rec["task"] is not None
            f.write(json.dumps(rec) + "\n")
    print(f"{args.n} requests in {time.monotonic() - t0:.0f}s: {n_ok} parsed, {args.n - n_ok} unparseable -> {out}")


if __name__ == "__main__":
    main()
