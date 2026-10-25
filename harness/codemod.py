"""Rule-based codemod baseline: what mechanical rewrite rules solve with no model at all.

Each rule is a context-free AST rewrite a careful engineer could apply blindly across a codebase. Types
that need understanding the code (DataFrame.append loops, positional Series indexing, groupby.apply
with grouping columns) get no rule, and neither do the held-out types, since a rule for them would
make the generalization comparison meaningless. The point of the baseline: GRPO's gain is only
interesting where it goes beyond this.

  codemod(source) -> rewritten source (unchanged if no rule applies)
"""
import ast

NP_ALIASES = {"float": "float", "int": "int", "bool": "bool", "object": "object", "str": "str",
              "complex": "complex", "long": "int", "unicode": "str"}
NP_RENAMES = {"product": "prod", "cumproduct": "cumprod", "alltrue": "all", "sometrue": "any",
              "trapz": "trapezoid", "in1d": "isin", "row_stack": "vstack"}
NP_CONSTANTS = {"NaN": "nan", "NAN": "nan", "Inf": "inf", "Infinity": "inf", "infty": "inf", "PINF": "inf"}
METHOD_RENAMES = {"iteritems": "items", "applymap": "map"}
INDEX_CLASSES = {"Int64Index": "int64", "UInt64Index": "uint64", "Float64Index": "float64"}
LEVEL_REDUCTIONS = {"sum", "mean", "max", "min", "count", "std", "var", "prod", "median", "any", "all", "sem"}
NUMERIC_ONLY_SAFE = {"sum", "prod", "mean", "median", "std", "var", "sem"}  # 1.x groupby default numeric_only=True for these (not min/max)
INPLACE_METHODS = {"fillna", "replace", "clip", "where", "mask", "interpolate", "ffill", "bfill"}


def _is_np(node, attr=None):
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "np" and (attr is None or node.attr == attr))


def _kw(call, name):
    return next((k for k in call.keywords if k.arg == name), None)


def _without(call, *names):
    return [k for k in call.keywords if k.arg not in names]


def _const(value):
    return ast.Constant(value=value)


class _Rewriter(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if _is_np(node) and node.attr in NP_ALIASES:                    # np.float -> float
            return ast.copy_location(ast.Name(id=NP_ALIASES[node.attr], ctx=ast.Load()), node)
        if _is_np(node) and node.attr in NP_CONSTANTS:                  # np.NaN -> np.nan
            node.attr = NP_CONSTANTS[node.attr]
        elif _is_np(node, "NINF"):                                      # np.NINF -> -np.inf
            return ast.copy_location(ast.UnaryOp(op=ast.USub(), operand=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr="inf", ctx=ast.Load())), node)
        elif _is_np(node) and node.attr in NP_RENAMES:                  # np.product -> np.prod
            node.attr = NP_RENAMES[node.attr]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if not isinstance(f, ast.Attribute):
            return node
        # .iteritems() / .applymap()
        if f.attr in METHOD_RENAMES:
            f.attr = METHOD_RENAMES[f.attr]
        # to_csv(line_terminator=) -> lineterminator=
        for k in node.keywords:
            if k.arg == "line_terminator":
                k.arg = "lineterminator"
        # .str.replace(a, b) -> keep the old regex=True default explicitly
        if f.attr == "replace" and isinstance(f.value, ast.Attribute) and f.value.attr == "str" \
                and _kw(node, "regex") is None and len(node.args) < 5:
            node.keywords.append(ast.keyword(arg="regex", value=_const(True)))
        # .fillna(method='ffill') -> .ffill()
        if f.attr == "fillna":
            m = _kw(node, "method")
            if m is not None and isinstance(m.value, ast.Constant) and not node.args:
                name = {"ffill": "ffill", "pad": "ffill", "bfill": "bfill", "backfill": "bfill"}.get(m.value.value)
                if name:
                    return ast.copy_location(ast.Call(func=ast.Attribute(value=f.value, attr=name, ctx=ast.Load()),
                                                      args=[], keywords=_without(node, "method")), node)
        # pd.Int64Index(x) -> pd.Index(x, dtype='int64')
        if isinstance(f.value, ast.Name) and f.value.id == "pd" and f.attr in INDEX_CLASSES:
            dtype = INDEX_CLASSES[f.attr]
            f.attr = "Index"
            if _kw(node, "dtype") is None:
                node.keywords.append(ast.keyword(arg="dtype", value=_const(dtype)))
        # x.ptp() -> np.ptp(x)
        if f.attr == "ptp" and not (isinstance(f.value, ast.Name) and f.value.id == "np"):
            return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                                                 attr="ptp", ctx=ast.Load()),
                                              args=[f.value] + node.args, keywords=node.keywords), node)
        # np.find_common_type(arrays, scalars) -> np.result_type(*arrays, *scalars)
        if _is_np(f, "find_common_type") and len(node.args) == 2:
            f.attr = "result_type"
            node.args = [ast.Starred(value=a, ctx=ast.Load()) for a in node.args]
        # np.array(x, copy=False) -> np.asarray(x)
        if _is_np(f, "array"):
            c = _kw(node, "copy")
            if c is not None and isinstance(c.value, ast.Constant) and c.value.value is False:
                f.attr = "asarray"
                node.keywords = _without(node, "copy")
        # pd.read_csv(..., squeeze=True) -> pd.read_csv(...).squeeze('columns')
        if f.attr == "read_csv":
            s = _kw(node, "squeeze")
            if s is not None and isinstance(s.value, ast.Constant) and s.value.value is True:
                node.keywords = _without(node, "squeeze")
                return ast.copy_location(ast.Call(func=ast.Attribute(value=node, attr="squeeze", ctx=ast.Load()),
                                                  args=[_const("columns")], keywords=[]), node)
        # s.sum(level=0) -> s.groupby(level=0, sort=False).sum()   (sum(level=) kept first-appearance order)
        if f.attr in LEVEL_REDUCTIONS and _kw(node, "level") is not None:
            lvl = _kw(node, "level")
            grp = ast.Call(func=ast.Attribute(value=f.value, attr="groupby", ctx=ast.Load()), args=[],
                           keywords=[ast.keyword(arg="level", value=lvl.value),
                                     ast.keyword(arg="sort", value=_const(False))])
            return ast.copy_location(ast.Call(func=ast.Attribute(value=grp, attr=f.attr, ctx=ast.Load()),
                                              args=node.args, keywords=_without(node, "level")), node)
        # .groupby(...) -> keep the old observed=False default explicitly (a no-op for non-categoricals)
        if f.attr == "groupby" and _kw(node, "observed") is None:
            node.keywords.append(ast.keyword(arg="observed", value=_const(False)))
        # df.groupby(...).mean() -> .mean(numeric_only=True)   (1.x silently dropped text columns)
        if f.attr in NUMERIC_ONLY_SAFE and _kw(node, "numeric_only") is None and isinstance(f.value, ast.Call) \
                and isinstance(f.value.func, ast.Attribute) and f.value.func.attr == "groupby":
            node.keywords.append(ast.keyword(arg="numeric_only", value=_const(True)))
        return node

    def visit_Expr(self, node):
        """df['a'].fillna(v, inplace=True) -> df['a'] = df['a'].fillna(v)"""
        self.generic_visit(node)
        c = node.value
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in INPLACE_METHODS \
                and isinstance(c.func.value, ast.Subscript):
            ip = _kw(c, "inplace")
            if ip is not None and isinstance(ip.value, ast.Constant) and ip.value.value is True:
                target = c.func.value
                store = ast.Subscript(value=target.value, slice=target.slice, ctx=ast.Store())
                call = ast.Call(func=c.func, args=c.args, keywords=_without(c, "inplace"))
                return ast.copy_location(ast.Assign(targets=[store], value=call), node)
        return node


def codemod(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    tree = ast.fix_missing_locations(_Rewriter().visit(tree))
    return ast.unparse(tree) + "\n"
