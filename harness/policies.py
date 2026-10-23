"""Policies the harness can evaluate. Anything with .name and .sample(messages, n, task) works.

  OpenAIPolicy   any OpenAI-compatible /chat/completions endpoint: a vLLM server for our checkpoints
                 or the base model, or a hosted frontier model as the reference ceiling
  ScriptedPolicy deterministic code-producing functions: the oracle (hand-written migration),
                 the unchanged legacy code, and the cheaters in harness/cheaters.py
"""
import json
import os
import time
import urllib.error
import urllib.request


def fence(code):
    return f"```python\n{code.strip()}\n```"


class ScriptedPolicy:
    deterministic = True

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def sample(self, messages, n, task):
        return [fence(self.fn(task))] * n


ORACLE = ScriptedPolicy("oracle", lambda t: t["modern"])
LEGACY = ScriptedPolicy("legacy", lambda t: t["legacy"])


class OpenAIPolicy:
    """One request per sample (some OpenAI-compatible servers ignore n>1). Retries 429/5xx."""
    deterministic = False

    def __init__(self, model, base_url, api_key_env="OPENAI_API_KEY", temperature=1.0,
                 max_tokens=1024, timeout_s=180, retries=5):
        self.name = f"openai:{model}"
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "EMPTY")      # vLLM accepts any key
        self.temperature, self.max_tokens = temperature, max_tokens
        self.timeout_s, self.retries = timeout_s, retries

    def _one(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
            time.sleep(min(60, 2 ** attempt))
        return ""

    def sample(self, messages, n, task):
        return [self._one(messages) for _ in range(n)]
