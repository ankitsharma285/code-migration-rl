"""Trajectory schema shared by evaluation and training.

An Episode is an ordered list of Steps. v1 (single-turn) episodes have exactly one step;
phase 2 (multi-turn agent) appends steps -- each tool call and its output -- with no format change.
Stored as JSON Lines: one episode per line.
"""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Step:
    prompt: list                 # chat messages shown to the policy at this step
    action: str                  # the policy's raw completion
    tool_output: str = ""        # what the environment returned to the policy (empty in v1)
    reward: float = 0.0
    reason: str = ""             # reward reason code (pass, ast, crash, warning, mismatch, ...)
    detail: str = ""
    timing: dict = field(default_factory=dict)   # e.g. {"generate_s", "score_wall_s", "startup_s"}


@dataclass
class Episode:
    task_id: str
    sample_idx: int
    policy: str
    hidden_seeds: list
    steps: list = field(default_factory=list)
    info: dict = field(default_factory=dict)    # v2 multi-turn: {"stop": ..., "turns": ...}

    @property
    def reward(self):            # v1: the final step's reward is the episode's reward
        return self.steps[-1].reward if self.steps else 0.0


def write_jsonl(path, episodes):
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep)) + "\n")


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["steps"] = [Step(**s) for s in d["steps"]]
            out.append(Episode(**d))
    return out
