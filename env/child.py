"""Sandboxed executor. Runs INSIDE a pinned runtime (legacy or modern).

Invoked by env/runner.py as:  <env python> -I env/child.py --mem-mb M --cpu-s S --exec-s W [--block-socket]
Reads one job as JSON on stdin:
    {"code": str, "entry": str, "inputs": [kwargs, ...], "nonce": str}
Writes one JSON object on stdout:
    {"nonce": str (echoed from the job; a candidate forging output cannot know it),
     "define": {"status", "detail", "warning", "warnings"},
     "calls": [{"status", "detail", "warning", "warnings", "result"}, ...]}
("warning" is the first deprecation warning; "warnings" lists every distinct one, even before a crash.)
Status values: OK, WARN (a deprecation-type warning was raised), FAIL (exception).
Warnings are recorded, not raised, so candidate control flow is identical in both runtimes;
the reward decides what a warning means (legacy: allowed; modern: reward 0).

Self-contained on purpose: -I keeps the repo and user site-packages off sys.path.
"""
import io
import json
import time
import math
import resource
import sys
import warnings

DEPRECATIONS = (DeprecationWarning, FutureWarning, PendingDeprecationWarning)
PRELUDE = "import io\nimport numpy as np\nimport pandas as pd\n"
MAX_DETAIL = 600          # long enough for pandas' own fix-it hints (used as execution feedback)


def set_limits(mem_mb):
    """Limits that apply from process start. Hard == soft, so the candidate cannot raise them."""
    mem = mem_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def start_execution_clock(cpu_s, exec_s):
    """Budget for the CANDIDATE only, started after imports so slow filesystems don't count.

    CPU: RLIMIT_CPU set to (CPU already used by imports) + cpu_s -> SIGXCPU kills the process.
    Wall: SIGALRM with its default action, which terminates the process; the candidate cannot
    catch it with try/except (the AST check bans the signal module, so it cannot install a handler).
    """
    import signal
    used = resource.getrusage(resource.RUSAGE_SELF)
    spent = int(used.ru_utime + used.ru_stime) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (spent + cpu_s, spent + cpu_s + 1))
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(exec_s)


def block_socket():
    """Fallback when network namespaces are unavailable. Weak alone; the AST check bans socket use."""
    import socket

    def refuse(*a, **k):
        raise OSError("network disabled in sandbox")
    socket.socket = refuse
    socket.create_connection = refuse


def plain(v):
    import numpy as np
    if isinstance(v, (list, tuple)):
        return [plain(i) for i in v]
    if isinstance(v, dict):
        return {str(k): plain(i) for k, i in v.items()}
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if v is None or isinstance(v, (bool, int, str)):
        return v
    return repr(v)


def ser(x):
    """Version-neutral serialization: values, dtypes, labels, shapes. Never pickle across runtimes."""
    import numpy as np
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return {"t": "df", "columns": plain(list(x.columns)), "index": plain(list(x.index)),
                "index_names": plain(list(x.index.names)), "dtypes": [str(d) for d in x.dtypes],
                "values": plain(x.values.tolist())}
    if isinstance(x, pd.Series):
        return {"t": "series", "name": plain(x.name), "index": plain(list(x.index)),
                "index_names": plain(list(x.index.names)), "dtype": str(x.dtype),
                "values": plain(x.tolist())}
    if isinstance(x, pd.Index):
        return {"t": "index", "name": plain(x.name), "dtype": str(x.dtype), "values": plain(list(x))}
    if isinstance(x, np.ndarray):
        return {"t": "ndarray", "dtype": str(x.dtype), "shape": list(x.shape), "values": plain(x.tolist())}
    if isinstance(x, np.generic):
        return {"t": "npscalar", "dtype": str(x.dtype), "value": plain(x.item())}
    if isinstance(x, np.dtype):
        return {"t": "dtype", "value": str(x)}
    if isinstance(x, (list, tuple)):
        return {"t": "list", "items": [ser(i) for i in x]}
    if isinstance(x, dict):
        return {"t": "dict", "items": {str(k): ser(v) for k, v in x.items()}}
    return {"t": type(x).__name__, "value": plain(x)}


def materialize(v):
    import pandas as pd
    if isinstance(v, dict) and "__df__" in v:
        return pd.DataFrame(v["__df__"])
    if isinstance(v, dict) and "__series__" in v:
        d = v["__series__"]
        idx = d.get("index")
        if idx and isinstance(idx[0], list):
            idx = pd.MultiIndex.from_tuples([tuple(i) for i in idx])
        return pd.Series(d["values"], index=idx, name=d.get("name"))
    return v


def all_deprecations(caught, limit=10):
    """Every distinct deprecation-type warning, in order (type detection needs all of them, not just one)."""
    seen = []
    for w in caught:
        if issubclass(w.category, DEPRECATIONS):
            msg = f"{w.category.__name__}: {w.message}"[:MAX_DETAIL]
            if msg not in seen:
                seen.append(msg)
    return seen[:limit]


def first_deprecation(caught):
    for w in caught:
        if issubclass(w.category, DEPRECATIONS):
            return f"{w.category.__name__}: {w.message}"[:MAX_DETAIL]
    return ""


def describe(e):
    return f"{type(e).__name__}: {e}"[:MAX_DETAIL]


def main():
    t_start = time.monotonic()
    args = sys.argv[1:]
    mem_mb = int(args[args.index("--mem-mb") + 1])
    cpu_s = int(args[args.index("--cpu-s") + 1])
    exec_s = int(args[args.index("--exec-s") + 1])
    set_limits(mem_mb)
    if "--block-socket" in args:
        block_socket()

    job = json.loads(sys.stdin.read())
    nonce = job.pop("nonce")             # proves the output came from this code, not the candidate
    protocol_out = sys.stdout
    sys.stdout = io.StringIO()          # candidate prints must not corrupt the protocol
    out = {"define": {}, "calls": [], "nonce": nonce}

    ns = {"__name__": "__candidate__"}
    exec(PRELUDE, ns)                    # imports happen before warning capture and before the clock
    out["startup_s"] = round(time.monotonic() - t_start, 3)
    start_execution_clock(cpu_s, exec_s)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            exec(compile(job["code"], "<candidate>", "exec"), ns)
            dep = first_deprecation(caught)
            out["define"] = {"status": "WARN" if dep else "OK", "detail": "", "warning": dep,
                             "warnings": all_deprecations(caught)}
        except BaseException as e:      # includes SystemExit from exit()
            out["define"] = {"status": "FAIL", "detail": describe(e), "warning": "",
                             "warnings": all_deprecations(caught)}

    fn = ns.get(job["entry"])
    if out["define"]["status"] != "FAIL" and not callable(fn):
        out["define"] = {"status": "FAIL", "detail": f"missing_entry: no function {job['entry']!r}",
                         "warning": ""}

    if out["define"]["status"] != "FAIL":
        for inp in job["inputs"]:
            kwargs = {k: materialize(v) for k, v in inp.items()}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = ser(fn(**kwargs))
                    dep = first_deprecation(caught)
                    out["calls"].append({"status": "WARN" if dep else "OK", "detail": "",
                                         "warning": dep, "warnings": all_deprecations(caught),
                                         "result": result})
                except BaseException as e:
                    out["calls"].append({"status": "FAIL", "detail": describe(e), "warning": "",
                                         "warnings": all_deprecations(caught),
                                         "result": None})

    protocol_out.write(json.dumps(out))
    protocol_out.flush()


if __name__ == "__main__":
    main()
