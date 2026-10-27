"""Reward: score a model completion for one migration task.

reward = 1.0 only if the completion
  - contains a ```python code block that parses,
  - passes the AST check (env/ast_check.py),
  - defines the task's entry function and runs in the MODERN runtime,
  - raises no deprecation-type warning (DeprecationWarning / FutureWarning) at definition or call time,
  - and matches the legacy reference output on every hidden input (env/compare.py rules).
Otherwise 0.0, or MALFORMED_PENALTY when there is no code block at all.
Every result carries a reason code for debugging and error analysis.
"""
import ast
import re
from dataclasses import dataclass, field

from env import ast_check as _ast
from env.compare import matches
from env.runner import Limits, run_job

MALFORMED_PENALTY = -0.1
REASONS = ("pass", "no_code", "syntax", "ast", "timeout", "crash", "missing_entry",
           "define_error", "warning", "mismatch")

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


@dataclass
class Reward:
    reward: float
    reason: str
    detail: str = ""
    per_input: list = field(default_factory=list)   # True/False per hidden input when it got that far
    wall_s: float = 0.0
    startup_s: float = 0.0


def extract_code(completion):
    """The LAST fenced code block (models often quote the old code first). None if there is none."""
    blocks = _FENCE.findall(completion)
    return blocks[-1] if blocks else None


def ast_check(code):
    """'' if allowed, else the first violation (env/ast_check.py lists the rules and why)."""
    violations = _ast.check(code)
    return violations[0] if violations else ""


def compute_references(task, inputs, limits=None):
    """Run the task's legacy code in the legacy runtime. Raises if legacy cannot produce references."""
    res = run_job("legacy", task["legacy"], task["entry"], inputs, limits)
    if res.status != "ok" or res.define.get("status") == "FAIL":
        raise RuntimeError(f"{task['id']}: legacy run failed: {res.status} {res.detail} {res.define}")
    bad = [c for c in res.calls if c["status"] == "FAIL"]
    if bad:
        raise RuntimeError(f"{task['id']}: legacy call failed: {bad[0]['detail']}")
    return [c["result"] for c in res.calls]


def references_on_valid_inputs(task, visible, candidates, n, limits=None):
    """Evaluation seeds are new inputs the task filter never ran, and a few produce inputs on which the LEGACY
    code itself fails (e.g. a NaN pattern the generator rarely makes). Such an input has no reference and
    cannot be scored. Keeps the first n candidate inputs on which the legacy code succeeds, in order (so a task
    with no bad input keeps exactly its first n seeds). Returns (visible refs, kept inputs, kept refs, n dropped)."""
    res = run_job("legacy", task["legacy"], task["entry"], visible + candidates, limits)
    if res.status != "ok" or res.define.get("status") == "FAIL":
        raise RuntimeError(f"{task['id']}: legacy run failed: {res.status} {res.detail} {res.define}")
    calls = res.calls
    nv = len(visible)
    if any(c["status"] == "FAIL" for c in calls[:nv]):
        raise RuntimeError(f"{task['id']}: legacy fails on a visible input")
    kept = [(inp, c["result"]) for inp, c in zip(candidates, calls[nv:]) if c["status"] != "FAIL"][:n]
    first_n_failures = sum(c["status"] == "FAIL" for c in calls[nv:nv + n])
    return [c["result"] for c in calls[:nv]], [i for i, _ in kept], [r for _, r in kept], first_n_failures


def score(task, completion, inputs, references, limits=None, check_ast=True):
    """check_ast=False exists only to test the sandbox and to measure what the AST check prevents."""
    code = extract_code(completion)
    if code is None:
        return Reward(MALFORMED_PENALTY, "no_code")
    try:
        ast.parse(code)
    except SyntaxError as e:
        return Reward(0.0, "syntax", f"line {e.lineno}: {e.msg}")
    violation = ast_check(code) if check_ast else ""
    if violation:
        return Reward(0.0, "ast", violation)

    res = run_job("modern", code, task["entry"], inputs, limits)
    if res.status == "timeout":
        return Reward(0.0, "timeout", res.detail, wall_s=res.wall_s)
    if res.status != "ok":
        return Reward(0.0, "crash", res.detail, wall_s=res.wall_s)

    d = res.define
    if d["status"] == "FAIL":
        reason = "missing_entry" if d["detail"].startswith("missing_entry") else "define_error"
        return Reward(0.0, reason, d["detail"], wall_s=res.wall_s)
    if d["status"] == "WARN":
        return Reward(0.0, "warning", d["warning"], wall_s=res.wall_s)

    per_input, first_problem = [], None
    for call, ref in zip(res.calls, references):
        if call["status"] == "FAIL":
            ok, problem = False, ("crash", call["detail"])
        elif call["status"] == "WARN":
            ok, problem = False, ("warning", call["warning"])
        else:
            ok = matches(ref, call["result"])
            problem = None if ok else ("mismatch", "output differs from legacy")
        per_input.append(ok)
        first_problem = first_problem or problem
    if first_problem:
        return Reward(0.0, first_problem[0], first_problem[1], per_input, res.wall_s, res.startup_s)
    return Reward(1.0, "pass", "", per_input, res.wall_s, res.startup_s)
