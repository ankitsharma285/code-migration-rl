"""Execution feedback: what a developer sees when they run code on the new library versions.

  feedback(task, code, visible_inputs, visible_refs) -> (text, all_ok)

Runs `code` in the MODERN runtime on the VISIBLE inputs only and reports, per example: the exception,
every deprecation warning, or a wrong output (shown next to the expected legacy output, which the
prompt already shows). The signature takes only visible inputs, so hidden inputs and their references
cannot reach the policy through this channel; harness/check_stage3.py audits it anyway.

Used two ways:
  - single-turn test: feedback on the LEGACY code, added to the first prompt (= turn 2 of a
    multi-turn episode whose turn 1 was "submit the code unchanged")
  - multi-turn agent: feedback on the model's previous attempt, as the environment's reply
"""
import ast
import json

from env import ast_check as _ast
from env.compare import matches
from env.runner import run_job

MAX_CHARS = 1500          # keeps a multi-turn episode inside the context budget
MAX_VALUE = 300


def _short(value):
    s = json.dumps(value, separators=(", ", ": "))
    return s if len(s) <= MAX_VALUE else s[:MAX_VALUE] + " ...(truncated)"


def feedback(task, code, visible_inputs, visible_refs, limits=None):
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}", False
    bad = _ast.check(code)
    if bad:
        return f"Rejected by the code checker: {bad[0]}", False

    res = run_job("modern", code, task["entry"], visible_inputs, limits)
    if res.status == "timeout":
        return "Timed out (the time limit is 10 seconds).", False
    if res.status != "ok":
        return f"The process crashed: {res.detail}", False

    lines, all_ok = [], True
    d = res.define
    for w in d.get("warnings", []):
        lines.append(f"While defining the function: {w}")
        all_ok = False
    if d["status"] == "FAIL":
        lines.append(f"Defining the function failed: {d['detail']}")
        return "\n".join(lines)[:MAX_CHARS], False

    seen, prev_error = set(), None
    for i, (call, ref) in enumerate(zip(res.calls, visible_refs), 1):
        warns = call.get("warnings", [])
        new = [w for w in warns if w not in seen]
        seen.update(new)
        for w in new:
            lines.append(f"Example {i}: {w}")
        if warns and not new:
            lines.append(f"Example {i}: the same warning(s) as above.")
        if warns:
            all_ok = False
        if call["status"] == "FAIL":
            same = call["detail"] == prev_error
            lines.append(f"Example {i}: raised the same error." if same else f"Example {i}: raised {call['detail']}")
            prev_error = call["detail"]
            all_ok = False
        elif not matches(ref, call["result"]):
            lines.append(f"Example {i}: wrong output.\n  expected: {_short(ref)}\n  got:      {_short(call['result'])}")
            all_ok = False
        elif not warns:
            lines.append(f"Example {i}: correct, no warnings.")
    return "\n".join(lines)[:MAX_CHARS], all_ok
