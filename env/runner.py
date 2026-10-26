"""Parent side of the sandbox: runs jobs in the pinned runtimes, one fresh subprocess per job.

Isolation per job:
  - fresh temp directory as cwd and HOME, deleted afterwards
  - memory (RLIMIT_AS), CPU-time, and file-size limits set inside the child (env/child.py)
  - wall-clock timeout; on expiry the whole process group is killed
  - no network: a user+network namespace (`unshare -rn`) when the machine allows it,
    otherwise a socket block inside the child (weaker; the AST check covers the gap)
  - in MODERN runs, the legacy env is hidden behind an empty tmpfs (mount namespace), so code
    cannot load the old pandas from disk even if it slips past the AST check
  - single-threaded math libraries (multithreaded BLAS in many parallel children exceeds
    per-user process limits and slows everything through contention)
"""
import json
import os
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHILD = REPO / "env" / "child.py"

SINGLE_THREAD = {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


@dataclass
class Limits:
    exec_s: int = 10            # wall clock for the candidate's code (all inputs), after imports
    cpu_s: int = 10             # CPU seconds for the candidate's code, after imports
    startup_s: float = 60.0     # importing pandas/NumPy; slow on network filesystems like Lustre
    mem_mb: int = 2048          # address space; pandas import alone needs a few hundred MB
    max_output_bytes: int = 5_000_000

    @property
    def total_s(self):          # parent-side safety net only; the child enforces exec_s itself
        return self.startup_s + self.exec_s + 5


@dataclass
class JobResult:
    status: str                 # "ok" | "timeout" | "crashed" | "bad_output"
    define: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    detail: str = ""
    wall_s: float = 0.0
    startup_s: float = 0.0      # child-reported import time (filesystem cost, not the candidate's)
    network_isolation: str = ""


def envs_dir():
    """CMRL_ENVS overrides where the pinned runtimes live (used when reproducing from a git worktree,
    which has the code at an old commit but no envs/ folder, since envs/ is not in git)."""
    return Path(os.environ.get("CMRL_ENVS", REPO / "envs"))


def python_for(runtime):
    return str(envs_dir() / runtime / "bin" / "python")


_HIDE = 'mount -t tmpfs tmpfs "$1" && shift && exec "$@"'


def _works(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@lru_cache(maxsize=1)
def isolation_mode():
    """Strongest isolation this machine allows, checked once:
      'namespace+hide-legacy'  no network, and the legacy env is invisible inside MODERN runs
                               (blocks loading the old pandas from disk, independent of the AST check)
      'namespace'              no network only
      'socket-block'           socket calls blocked inside the child (weakest; the AST check covers it)
    """
    if not shutil.which("unshare"):
        return "socket-block"
    probe_dir = tempfile.mkdtemp(prefix="sbx_probe_")
    try:
        if _works(["unshare", "-rmn", "sh", "-c", _HIDE, "sh", probe_dir, "true"]):
            return "namespace+hide-legacy"
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    return "namespace" if _works(["unshare", "-rn", "true"]) else "socket-block"


def network_mode():                     # kept for older callers
    return isolation_mode()


_SLOTS = None


def set_max_concurrent(n):
    """Cap simultaneous sandbox processes across threads (multi-turn eval runs many episodes at once)."""
    global _SLOTS
    _SLOTS = threading.BoundedSemaphore(n) if n else None


def run_job(runtime, code, entry, inputs, limits=None):
    """Run `code` in `runtime` ('legacy' or 'modern'), call `entry(**kw)` for each kw in `inputs`."""
    if _SLOTS is None:
        return _run_job(runtime, code, entry, inputs, limits)
    with _SLOTS:
        return _run_job(runtime, code, entry, inputs, limits)


def _run_job(runtime, code, entry, inputs, limits=None):
    limits = limits or Limits()
    mode = isolation_mode()
    cmd = [python_for(runtime), "-I", str(CHILD), "--mem-mb", str(limits.mem_mb), "--cpu-s", str(limits.cpu_s),
           "--exec-s", str(limits.exec_s)]
    if mode == "namespace+hide-legacy" and runtime == "modern":
        legacy_dir = str((envs_dir() / "legacy").resolve())
        cmd = ["unshare", "-rmn", "sh", "-c", _HIDE, "sh", legacy_dir] + cmd
    elif mode.startswith("namespace"):
        cmd = ["unshare", "-rn"] + cmd
    else:
        cmd.append("--block-socket")

    workdir = tempfile.mkdtemp(prefix="sbx_")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": workdir, "TMPDIR": workdir,
           "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", **SINGLE_THREAD}
    nonce = secrets.token_hex(16)
    payload = json.dumps({"code": code, "entry": entry, "inputs": inputs, "nonce": nonce})
    start = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=workdir, env=env, start_new_session=True, text=True)
        try:
            out, err = proc.communicate(payload, timeout=limits.total_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return JobResult("timeout", detail=f"exceeded {limits.total_s:.0f}s including startup",
                             wall_s=time.monotonic() - start, network_isolation=mode)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    wall = time.monotonic() - start

    if proc.returncode == -signal.SIGXCPU:
        return JobResult("timeout", detail=f"exceeded {limits.cpu_s}s CPU", wall_s=wall, network_isolation=mode)
    if proc.returncode == -signal.SIGALRM:
        return JobResult("timeout", detail=f"exceeded {limits.exec_s}s wall clock", wall_s=wall,
                         network_isolation=mode)
    if proc.returncode != 0 or not out:
        sig = -proc.returncode if proc.returncode and proc.returncode < 0 else None
        why = f"signal {signal.Signals(sig).name}" if sig else f"exit code {proc.returncode}"
        tail = (err or "").strip().splitlines()[-1:] or [""]
        return JobResult("crashed", detail=f"{why}: {tail[0][:200]}", wall_s=wall, network_isolation=mode)
    if len(out) > limits.max_output_bytes:
        return JobResult("bad_output", detail="output too large", wall_s=wall, network_isolation=mode)
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return JobResult("bad_output", detail="unparseable child output", wall_s=wall, network_isolation=mode)
    if not isinstance(data, dict) or data.get("nonce") != nonce:
        return JobResult("bad_output", detail="output not produced by the sandbox (nonce mismatch)",
                         wall_s=wall, network_isolation=mode)
    return JobResult("ok", define=data["define"], calls=data["calls"], wall_s=wall,
                     startup_s=data.get("startup_s", 0.0), network_isolation=mode)


def run_many(jobs, workers=None, limits=None):
    """jobs: list of (runtime, code, entry, inputs). Returns JobResults in the same order."""
    # sched_getaffinity respects the cores Slurm actually allocated; cpu_count() reports the whole node
    workers = workers or max(1, len(os.sched_getaffinity(0)) - 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: run_job(*j, limits=limits), jobs))
