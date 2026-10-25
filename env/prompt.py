"""The observation the policy sees. Leakage rule: ONLY the legacy function and the visible examples
(inputs from VISIBLE_SEEDS and the legacy outputs on them). Hidden inputs and their reference
outputs never appear; harness/check_stage3.py audits this for every task.
"""
import json

SYSTEM = ("You migrate Python code from pandas 1.3 / NumPy 1.21 to pandas 2.2 / NumPy 2.x. "
          "The migrated code must behave identically: same outputs, dtypes, and names.")

TEMPLATE = """Migrate this function so it runs on pandas 2.2 and NumPy 2.x with no DeprecationWarning or FutureWarning, and returns exactly what it returns today.

```python
{legacy}
```

`pd`, `np`, and `io` are already imported. Keep the function name `{entry}` and its arguments.

{examples}

Reply with the complete migrated function in a single ```python code block. Do not call the function or add example usage."""

FEEDBACK_TEMPLATE = """

Running the current code on pandas 2.2 / NumPy 2.x with the examples above gives:
```text
{feedback}
```"""


def _show(value, limit=600):
    s = json.dumps(value, separators=(", ", ": "))
    return s if len(s) <= limit else s[:limit] + " ...(truncated)"


def build(task, visible_inputs, visible_outputs, feedback=None):
    """visible_outputs: serialized legacy results for visible_inputs (from compute_references).
    feedback: optional env/feedback.py text for the legacy code, appended after the task."""
    ex = []
    for i, (inp, out) in enumerate(zip(visible_inputs, visible_outputs), 1):
        ex.append(f"Example {i}\n  arguments: {_show(inp)}\n  returns: {_show(out)}")
    examples = ("Arguments use {\"__df__\": ...} for a DataFrame and {\"__series__\": ...} for a Series.\n\n"
                + "\n\n".join(ex))
    user = TEMPLATE.format(legacy=task["legacy"].strip(), entry=task["entry"], examples=examples)
    if feedback is not None:
        user += FEEDBACK_TEMPLATE.format(feedback=feedback)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# Few-shot baseline only (GRPO never sees demos). Hand-written seed tasks, training types only, ordered so
# the first demos cover the migrations no rename fixes (append loops, positional indexing, apply on groups).
FEWSHOT_ORDER = ["s01_group_summary", "s09_fill_and_ends", "s10_growth_factor", "s17_nested_totals",
                 "s20_return_stats", "s02_column_totals"]
DEMO_TEMPLATE = """Migrate this function so it runs on pandas 2.2 and NumPy 2.x with no DeprecationWarning or FutureWarning, and returns exactly what it returns today.

```python
{legacy}
```

Reply with the complete migrated function in a single ```python code block."""


def with_fewshot(messages, task, k):
    """Insert k (legacy -> modern) demo turns between the system and user messages. Demos are short (no
    visible examples) to stay inside the 4096-token context; a task is never its own demo."""
    if not k:
        return messages
    from tasks.seed_tasks import SEED_TASKS
    by_id = {t["id"]: t for t in SEED_TASKS}
    demos = [by_id[i] for i in FEWSHOT_ORDER if i != task["id"]][:k]
    turns = []
    for d in demos:
        turns.append({"role": "user", "content": DEMO_TEMPLATE.format(legacy=d["legacy"].strip())})
        turns.append({"role": "assistant", "content": f"```python\n{d['modern'].strip()}\n```"})
    return messages[:1] + turns + messages[1:]
