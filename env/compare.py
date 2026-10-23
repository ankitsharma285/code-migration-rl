"""Comparator: does a candidate's serialized output match the legacy reference?

Rules (decided 2026-10-22, see the plan's Output comparison section):
  - values, shapes, and labels must match; floats within rtol 1e-6 / atol 1e-9; NaN == NaN
  - dtypes match by KIND (int/float/bool/...), so float64 vs float32 is a match (NEP 50)
  - names count as behavior: Series names, index names, column labels must match
  - object columns holding numbers are not reconciled here; such tasks are filtered out upstream
"""
import math

RTOL, ATOL = 1e-6, 1e-9
_KINDS = (("uint", "u"), ("int", "i"), ("float", "f"), ("bool", "b"),
          ("complex", "c"), ("datetime", "M"), ("timedelta", "m"))


def dtype_kind(dt):
    """'int64' -> 'i', 'Int64' (nullable) -> 'i', 'UInt32' -> 'u', 'float32' -> 'f'; else unchanged."""
    s = str(dt).lower()
    for prefix, k in _KINDS:
        if s.startswith(prefix):
            return k
    return str(dt)


def same(a, b, mode="kind"):
    """mode: 'exact' compares dtype strings, 'kind' compares dtype kinds, 'values' ignores dtypes."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        for k in a:
            if k in ("dtype", "dtypes") and mode != "exact":
                if mode == "values":
                    continue
                av = a[k] if isinstance(a[k], list) else [a[k]]
                bv = b[k] if isinstance(b[k], list) else [b[k]]
                if [dtype_kind(x) for x in av] != [dtype_kind(x) for x in bv]:
                    return False
            elif not same(a[k], b[k], mode):
                return False
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, mode) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def matches(reference, candidate):
    return same(reference, candidate, "kind")
