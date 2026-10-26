"""Multi-turn episode with automatic execution (v2).

  turn 1 : the task prompt + execution feedback on the LEGACY code (env/feedback.py)
  turn k : the policy replies with code -> the sandbox runs it on the VISIBLE inputs -> feedback
  stop   : the visible examples pass with no warnings ("visible_pass"), or max_turns is reached,
           or the reply has no code block twice in a row, or the context budget runs out
  reward : the LAST attempt, scored on the HIDDEN inputs (env/reward.py), exactly as in v1

Every attempt is also scored on the hidden inputs for analysis (pass_by_turn), never shown to the
policy. The policy only ever sees visible inputs, their legacy outputs, and feedback computed from
them, so the feedback channel cannot leak hidden data (harness/check_multiturn.py audits this).

Kept as act -> observe so a v3 can add explicit actions (run / submit) without rewriting the loop.
"""
from dataclasses import dataclass, field

from env import prompt as prompt_mod
from env.feedback import feedback
from env.reward import extract_code, score

FOLLOWUP = """Running your code on pandas 2.2 / NumPy 2.x with the examples gives:
```text
{feedback}
```

Fix the problems and reply with the complete migrated function in a single ```python code block."""
NO_CODE = "No ```python code block was found in your reply."


class ContextFull(Exception):
    """Raised by a generate() callable when the conversation no longer fits the model's context."""


@dataclass
class Turn:
    messages: list            # what the policy saw at this turn
    completion: str
    feedback: str             # environment reply on the VISIBLE inputs (shown at the next turn)
    visible_ok: bool
    hidden: object = None     # env.reward.Reward of this attempt on the HIDDEN inputs (analysis only)


@dataclass
class EpisodeResult:
    turns: list = field(default_factory=list)
    stop: str = ""            # visible_pass | max_turns | no_code | context
    final: object = None      # Reward of the last attempt (= the episode's reward)


def run_episode(task, generate, vis, vis_refs, hid, hid_refs, max_turns=3, limits=None, score_every_turn=True,
                first_feedback=None):
    """generate(messages) -> completion text (one sample). Raises ContextFull when out of context.
    first_feedback: the legacy code's feedback, if already computed (same for every sample of a task)."""
    first_fb = first_feedback if first_feedback is not None else feedback(task, task["legacy"], vis, vis_refs, limits)[0]
    messages = prompt_mod.build(task, vis, vis_refs, first_fb)
    res = EpisodeResult()
    for k in range(1, max_turns + 1):
        try:
            completion = generate(messages)
        except ContextFull:
            res.stop = "context"
            break
        code = extract_code(completion)
        if code is None:
            fb, ok = NO_CODE, False
        else:
            fb, ok = feedback(task, code, vis, vis_refs, limits)
        turn = Turn(list(messages), completion, fb, ok)
        if score_every_turn or ok or k == max_turns:
            turn.hidden = score(task, completion, hid, hid_refs, limits)
        res.turns.append(turn)
        if ok:
            res.stop = "visible_pass"
            break
        if code is None and len(res.turns) >= 2 and extract_code(res.turns[-2].completion) is None:
            res.stop = "no_code"
            break
        messages = messages + [{"role": "assistant", "content": completion},
                               {"role": "user", "content": FOLLOWUP.format(feedback=fb)}]
    else:
        res.stop = "max_turns"
    if res.turns:
        last = res.turns[-1]
        if last.hidden is None:
            last.hidden = score(task, last.completion, hid, hid_refs, limits)
        res.final = last.hidden
    return res
