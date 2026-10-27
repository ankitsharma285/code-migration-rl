"""Static check on candidate code, run BEFORE execution. Rejects code that could fool the reward.

Design: allow-list imports (everything else is banned by default) plus a few targeted bans.
Each rule closes a concrete route to a reward of 1 without a real migration:

  imports outside ALLOWED_MODULES  -> warnings (suppress the warning check), sys/importlib (load the
                                      legacy pandas from disk), gc/inspect (read the sandbox nonce
                                      and forge output), os/signal/socket/subprocess (escape limits)
  private / internal submodules    -> pandas.core, numpy._core etc. expose removed APIs under new names
  attributes starting with "_"     -> df._append (removed API kept privately), __class__/__dict__ tricks
  getattr/eval/exec/open/...       -> string-built attribute access and dynamic code bypass the above
  io.open / io.FileIO              -> the same file access as open() through an allowed module
  bare / broad except clauses      -> swallowing the error a removed API raises
  assigning to pd./np. attributes  -> monkeypatching a removed API back in, or flipping library modes
  set_option / option_context      -> changing global library behavior instead of migrating code

Returns a list of violation strings; empty means allowed.
"""
import ast

ALLOWED_MODULES = {
    "pandas", "numpy", "io", "math", "re", "json", "datetime", "collections", "itertools",
    "functools", "statistics", "decimal", "fractions", "string", "operator", "typing", "copy",
    "bisect", "heapq",
}
INTERNAL_SUBMODULES = {"core", "_core", "_libs", "compat", "testing", "lib"}
BANNED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "globals",
    "locals", "vars", "dir", "breakpoint", "input", "exit", "quit", "help", "memoryview",
}
BROAD_EXCEPTIONS = {
    "Exception", "BaseException", "Warning", "DeprecationWarning", "FutureWarning",
    "PendingDeprecationWarning", "UserWarning", "RuntimeWarning",
}
FILE_ATTRS = {"open", "open_code", "FileIO"}
OPTION_CALLS = {"set_option", "option_context", "reset_option", "seterr", "seterrcall"}
LIBRARY_ROOTS = {"pandas", "numpy"}


class _Checker(ast.NodeVisitor):
    def __init__(self):
        self.violations = []
        self.lib_names = {"pd", "np", "pandas", "numpy"}   # the sandbox prelude binds pd and np

    def flag(self, node, msg):
        self.violations.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def _check_module(self, node, name):
        parts = name.split(".")
        if parts[0] not in ALLOWED_MODULES:
            self.flag(node, f"import of {parts[0]!r} is not allowed")
        elif parts[0] in LIBRARY_ROOTS and any(p.startswith("_") or p in INTERNAL_SUBMODULES for p in parts[1:]):
            self.flag(node, f"import of internal module {name!r}")

    def visit_Import(self, node):
        for a in node.names:
            self._check_module(node, a.name)
            if a.name.split(".")[0] in LIBRARY_ROOTS:
                self.lib_names.add((a.asname or a.name).split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level or not node.module:
            self.flag(node, "relative import")
        else:
            self._check_module(node, node.module)
        for a in node.names:
            if a.name == "*":
                self.flag(node, "star import")
            elif a.name.startswith("_"):
                self.flag(node, f"import of private name {a.name!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            self.flag(node, f"access to private/dunder attribute {node.attr!r}")
        if node.attr in FILE_ATTRS and self._root_name(node) == "io":
            self.flag(node, f"file access via io.{node.attr}")
        if node.attr in OPTION_CALLS:
            self.flag(node, f"global library option change via {node.attr!r}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in BANNED_NAMES:
            self.flag(node, f"use of {node.id!r}")
        elif node.id.startswith("__"):
            self.flag(node, f"use of dunder name {node.id!r}")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.flag(node, "bare except")
        else:
            names = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            for n in names:
                if isinstance(n, ast.Name) and n.id in BROAD_EXCEPTIONS:
                    self.flag(node, f"broad except {n.id}")
        self.generic_visit(node)

    def _root_name(self, node):
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def _check_targets(self, targets):
        """Flag `pd.X = ...` / `np.a.b[...] = ...`, but not `arr[np.isinf(arr)] = ...` (index, not target)."""
        for t in targets:
            if isinstance(t, (ast.Tuple, ast.List)):
                self._check_targets(t.elts)
                continue
            node = t
            while isinstance(node, ast.Subscript):     # what is being written is the subscripted object
                node = node.value
            if isinstance(node, ast.Attribute) and self._root_name(node) in self.lib_names:
                self.flag(node, f"assignment to library attribute ({self._root_name(node)}.{node.attr})")

    def visit_Assign(self, node):
        self._check_targets(node.targets)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._check_targets([node.target])
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._check_targets([node.target])
        self.generic_visit(node)

    def visit_Delete(self, node):
        self._check_targets(node.targets)
        self.generic_visit(node)


def check(code):
    """List of violations ('' entries never). Assumes the code already parses."""
    checker = _Checker()
    checker.visit(ast.parse(code))
    return checker.violations
