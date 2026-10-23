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

Reply with the complete migrated function in a single ```python code block."""


def _show(value, limit=600):
    s = json.dumps(value, separators=(", ", ": "))
    return s if len(s) <= limit else s[:limit] + " ...(truncated)"


def build(task, visible_inputs, visible_outputs):
    """visible_outputs: serialized legacy results for visible_inputs (from compute_references)."""
    ex = []
    for i, (inp, out) in enumerate(zip(visible_inputs, visible_outputs), 1):
        ex.append(f"Example {i}\n  arguments: {_show(inp)}\n  returns: {_show(out)}")
    examples = ("Arguments use {\"__df__\": ...} for a DataFrame and {\"__series__\": ...} for a Series.\n\n"
                + "\n\n".join(ex))
    user = TEMPLATE.format(legacy=task["legacy"].strip(), entry=task["entry"], examples=examples)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
