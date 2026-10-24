"""Hand-written seed tasks: ground truth for testing the sandbox, comparator, and cheater suite.

Each task has:
  id        unique name
  types     migration types it exercises (training types only; see categories.json)
  entry     name of the function to call
  legacy    source written against pandas 1.3 / NumPy 1.21
  modern    a correct hand-written migration for pandas 2.2 / NumPy 2.x
  gen       source defining gen(rng) -> kwargs, where rng is random.Random(seed).
            Values are plain JSON data; tagged dicts become pandas objects in each runtime:
              {"__df__": {col: [...]}}                         -> pd.DataFrame
              {"__series__": {"values": [...], "index": [...], "name": ...}} -> pd.Series
              (an index whose items are lists becomes a MultiIndex)
            Inputs are generated once, outside both runtimes, so legacy and modern see
            identical data (NumPy's Generator streams are not guaranteed stable across versions).
"""

SEED_TASKS = [
    dict(
        id="s01_group_summary",
        types=["df_append_loop"],
        entry="summarize",
        legacy='''
def summarize(df):
    result = None
    for g in sorted(df['g'].unique()):
        sub = df[df['g'] == g]
        row = pd.DataFrame({'g': [g], 'total': [sub['v'].sum()], 'n': [len(sub)]})
        result = row if result is None else result.append(row, ignore_index=True)
    return result
''',
        modern='''
def summarize(df):
    rows = []
    for g in sorted(df['g'].unique()):
        sub = df[df['g'] == g]
        rows.append(pd.DataFrame({'g': [g], 'total': [sub['v'].sum()], 'n': [len(sub)]}))
    return pd.concat(rows, ignore_index=True)
''',
        gen='''
def gen(rng):
    n = rng.randint(5, 15)
    return {"df": {"__df__": {"g": [rng.choice("abcd") for _ in range(n)],
                              "v": [rng.randint(-20, 50) for _ in range(n)]}}}
''',
    ),
    dict(
        id="s02_column_totals",
        types=["series_append", "iteritems"],
        entry="column_totals",
        legacy='''
def column_totals(df):
    totals = pd.Series(dtype='float64')
    for name, col in df.iteritems():
        totals = totals.append(pd.Series([col.sum()], index=[name]))
    return totals
''',
        modern='''
def column_totals(df):
    return pd.Series({name: col.sum() for name, col in df.items()}, dtype='float64')
''',
        gen='''
def gen(rng):
    n = rng.randint(3, 10)
    cols = rng.sample(["price", "tax", "fee", "discount", "tip"], rng.randint(2, 4))
    return {"df": {"__df__": {c: [round(rng.uniform(0, 100), 2) for _ in range(n)] for c in cols}}}
''',
    ),
    dict(
        id="s03_scale_cells",
        types=["applymap", "np_type_aliases"],
        entry="scale_cells",
        legacy='''
def scale_cells(df):
    return df.applymap(lambda x: np.float(x) / 100)
''',
        modern='''
def scale_cells(df):
    return df.map(lambda x: float(x) / 100)
''',
        gen='''
def gen(rng):
    n = rng.randint(2, 6)
    return {"df": {"__df__": {c: [rng.randint(0, 500) for _ in range(n)] for c in ["q1", "q2", "q3"]}}}
''',
    ),
    dict(
        id="s04_group_means_ffill",
        types=["groupby_numeric_only", "fillna_method"],
        entry="group_means",
        legacy='''
def group_means(df):
    filled = df.fillna(method='ffill')
    return filled.groupby('k').mean()
''',
        modern='''
def group_means(df):
    filled = df.ffill()
    return filled.groupby('k').mean(numeric_only=True)
''',
        gen='''
def gen(rng):
    n = rng.randint(4, 12)
    v = [round(rng.uniform(0, 10), 3) for _ in range(n)]
    for i in range(1, n):
        if rng.random() < 0.3:
            v[i] = None
    v[rng.randrange(1, n)] = None  # always at least one gap, so the fill direction matters
    return {"df": {"__df__": {"k": [rng.choice("xyz") for _ in range(n)], "v": v,
                              "note": [rng.choice(["ok", "late", "void"]) for _ in range(n)]}}}
''',
    ),
    dict(
        id="s05_region_totals",
        types=["level_reduction"],
        entry="region_totals",
        legacy='''
def region_totals(s):
    return pd.DataFrame({'total': s.sum(level=0), 'peak': s.max(level=0)})
''',
        modern='''
def region_totals(s):
    g = s.groupby(level=0, sort=False)
    return pd.DataFrame({'total': g.sum(), 'peak': g.max()})
''',
        gen='''
def gen(rng):
    pairs = sorted({(rng.choice(["east", "west", "north"]), rng.randint(1, 9)) for _ in range(rng.randint(4, 10))})
    rng.shuffle(pairs)   # unsorted on purpose: groupby(level=0) without sort=False would reorder
    return {"s": {"__series__": {"values": [rng.randint(0, 100) for _ in pairs],
                                 "index": [list(p) for p in pairs], "name": "sales"}}}
''',
    ),
    dict(
        id="s06_parse_ids",
        types=["read_csv_squeeze", "str_replace_regex"],
        entry="parse_ids",
        legacy='''
def parse_ids(text):
    ids = pd.read_csv(io.StringIO(text), squeeze=True)
    return ids.str.replace('[^0-9]', '')
''',
        modern='''
def parse_ids(text):
    ids = pd.read_csv(io.StringIO(text)).squeeze('columns')
    return ids.str.replace('[^0-9]', '', regex=True)
''',
        gen='''
def gen(rng):
    rows = [f"{rng.choice('ABCD')}-{rng.randint(1, 999)}{rng.choice(['', 'x', 'q'])}"
            for _ in range(rng.randint(2, 8))]
    return {"text": "id\\n" + "\\n".join(rows) + "\\n"}
''',
    ),
    dict(
        id="s07_clean_phone",
        types=["str_replace_regex"],
        entry="clean_phone",
        legacy='''
def clean_phone(s):
    return s.str.replace(r'[\\s\\-\\(\\)]', '')
''',
        modern='''
def clean_phone(s):
    return s.str.replace(r'[\\s\\-\\(\\)]', '', regex=True)
''',
        gen='''
def gen(rng):
    def phone():
        a, b, c = rng.randint(200, 999), rng.randint(100, 999), rng.randint(0, 9999)
        return rng.choice([f"({a}) {b}-{c:04d}", f"{a}-{b}-{c:04d}", f"{a} {b} {c:04d}"])
    return {"s": {"__series__": {"values": [phone() for _ in range(rng.randint(2, 6))], "name": "phone"}}}
''',
    ),
    dict(
        id="s08_sales_by_size",
        types=["groupby_observed", "int64index"],
        entry="sales_by_size",
        legacy='''
def sales_by_size(df):
    cat = pd.Categorical(df['size'], categories=['S', 'M', 'L', 'XL'])
    totals = df.groupby(cat)['qty'].sum()
    positions = pd.Int64Index(range(len(totals)))
    return pd.Series(totals.values, index=positions, name='qty')
''',
        modern='''
def sales_by_size(df):
    cat = pd.Categorical(df['size'], categories=['S', 'M', 'L', 'XL'])
    totals = df.groupby(cat, observed=False)['qty'].sum()
    positions = pd.Index(range(len(totals)), dtype='int64')
    return pd.Series(totals.values, index=positions, name='qty')
''',
        gen='''
def gen(rng):
    n = rng.randint(3, 10)
    sizes = rng.sample(["S", "M", "L", "XL"], rng.randint(1, 3))
    return {"df": {"__df__": {"size": [rng.choice(sizes) for _ in range(n)],
                              "qty": [rng.randint(1, 20) for _ in range(n)]}}}
''',
    ),
    dict(
        id="s09_fill_and_ends",
        types=["chained_inplace", "series_positional_getitem"],
        entry="fill_and_ends",
        legacy='''
def fill_and_ends(df):
    df = df.copy()
    df['score'].fillna(df['score'].mean(), inplace=True)
    s = df.set_index('name')['score']
    return {'first': s[0], 'last': s[-1], 'mean': s.mean()}
''',
        modern='''
def fill_and_ends(df):
    df = df.copy()
    df['score'] = df['score'].fillna(df['score'].mean())
    s = df.set_index('name')['score']
    return {'first': s.iloc[0], 'last': s.iloc[-1], 'mean': s.mean()}
''',
        gen='''
def gen(rng):
    n = rng.randint(3, 8)
    score = [round(rng.uniform(50, 100), 1) for _ in range(n)]
    for i in range(n):
        if rng.random() < 0.3:
            score[i] = None
    i, j = rng.sample(range(n), 2)
    score[i], score[j] = 75.0, None  # always one known value and one gap to fill
    names = rng.sample(["ana", "ben", "cai", "dev", "eli", "fay", "gus", "hal"], n)
    return {"df": {"__df__": {"name": names, "score": score}}}
''',
    ),
    dict(
        id="s10_growth_factor",
        types=["groupby_apply_groups", "np_product_family"],
        entry="growth_factor",
        legacy='''
def growth_factor(df):
    return df.groupby('k').apply(lambda g: np.product(1 + g['r'].values))
''',
        modern='''
def growth_factor(df):
    return df.groupby('k').apply(lambda g: np.prod(1 + g['r'].values), include_groups=False)
''',
        gen='''
def gen(rng):
    n = rng.randint(3, 12)
    return {"df": {"__df__": {"k": [rng.choice("abc") for _ in range(n)],
                              "r": [round(rng.uniform(-0.1, 0.2), 4) for _ in range(n)]}}}
''',
    ),
    dict(
        id="s11_capped_ratios",
        types=["np_type_aliases", "np_nan_inf_aliases"],
        entry="capped_ratios",
        legacy='''
def capped_ratios(num, den):
    num = np.asarray(num, dtype=np.float)
    den = np.asarray(den, dtype=np.float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / den
    r[np.isinf(r)] = np.NaN
    return np.where(r > 10, np.Inf, r)
''',
        modern='''
def capped_ratios(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / den
    r[np.isinf(r)] = np.nan
    return np.where(r > 10, np.inf, r)
''',
        gen='''
def gen(rng):
    n = rng.randint(3, 8)
    return {"num": [rng.randint(0, 50) for _ in range(n)],
            "den": [rng.choice([0, 1, 2, 3, 5]) for _ in range(n)]}
''',
    ),
    dict(
        id="s12_curve_areas",
        types=["np_row_stack", "np_trapz"],
        entry="curve_areas",
        legacy='''
def curve_areas(curves):
    m = np.row_stack(curves)
    return np.array([np.trapz(row) for row in m])
''',
        modern='''
def curve_areas(curves):
    m = np.vstack(curves)
    return np.array([np.trapezoid(row) for row in m])
''',
        gen='''
def gen(rng):
    k = rng.randint(3, 7)
    return {"curves": [[round(rng.uniform(0, 5), 2) for _ in range(k)] for _ in range(rng.randint(1, 4))]}
''',
    ),
    dict(
        id="s13_selected_spread",
        types=["np_in1d", "ndarray_ptp"],
        entry="selected_spread",
        legacy='''
def selected_spread(values, wanted):
    a = np.asarray(values)
    sel = a[np.in1d(a, wanted)]
    return sel.ptp() if sel.size else 0
''',
        modern='''
def selected_spread(values, wanted):
    a = np.asarray(values)
    sel = a[np.isin(a, wanted)]
    return np.ptp(sel) if sel.size else 0
''',
        gen='''
def gen(rng):
    values = [rng.randint(0, 20) for _ in range(rng.randint(3, 12))]
    return {"values": values, "wanted": rng.sample(range(21), rng.randint(1, 8))}
''',
    ),
    dict(
        id="s14_stack_columns",
        types=["array_copy_false", "find_common_type"],
        entry="stack_columns",
        legacy='''
def stack_columns(cols):
    arrays = [np.array(c, copy=False) for c in cols]
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    return np.concatenate([a.astype(dtype) for a in arrays])
''',
        modern='''
def stack_columns(cols):
    arrays = [np.asarray(c) for c in cols]
    dtype = np.result_type(*[a.dtype for a in arrays])
    return np.concatenate([a.astype(dtype) for a in arrays])
''',
        gen='''
def gen(rng):
    def col():
        if rng.random() < 0.5:
            return [rng.randint(-5, 5) for _ in range(rng.randint(1, 4))]
        return [round(rng.uniform(-5, 5), 2) for _ in range(rng.randint(1, 4))]
    return {"cols": [col() for _ in range(rng.randint(2, 4))]}
''',
    ),
    dict(
        id="s15_csv_report",
        types=["to_csv_line_terminator", "iteritems"],
        entry="csv_report",
        legacy='''
def csv_report(df):
    lines = [f"{name}: max={col.max()}" for name, col in df.iteritems()]
    buf = io.StringIO()
    df.to_csv(buf, index=False, line_terminator='\\n')
    return buf.getvalue() + '\\n'.join(lines)
''',
        modern='''
def csv_report(df):
    lines = [f"{name}: max={col.max()}" for name, col in df.items()]
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator='\\n')
    return buf.getvalue() + '\\n'.join(lines)
''',
        gen='''
def gen(rng):
    n = rng.randint(2, 5)
    return {"df": {"__df__": {"units": [rng.randint(0, 99) for _ in range(n)],
                              "stock": [rng.randint(0, 999) for _ in range(n)]}}}
''',
    ),
    dict(
        id="s16_running_log",
        types=["df_append_loop", "fillna_method"],
        entry="running_log",
        legacy='''
def running_log(events):
    events = events.fillna(method='bfill')
    log = pd.DataFrame({'step': [0], 'cum': [0.0]})
    total = 0.0
    for i, v in enumerate(events['val']):
        total += v
        log = log.append(pd.DataFrame({'step': [i + 1], 'cum': [total]}), ignore_index=True)
    return log
''',
        modern='''
def running_log(events):
    events = events.bfill()
    parts = [pd.DataFrame({'step': [0], 'cum': [0.0]})]
    total = 0.0
    for i, v in enumerate(events['val']):
        total += v
        parts.append(pd.DataFrame({'step': [i + 1], 'cum': [total]}))
    return pd.concat(parts, ignore_index=True)
''',
        gen='''
def gen(rng):
    n = rng.randint(2, 8)
    val = [round(rng.uniform(0, 10), 2) for _ in range(n)]
    for i in range(n - 1):
        if rng.random() < 0.3:
            val[i] = None
    val[rng.randrange(0, n - 1)] = None  # always at least one gap, so the fill direction matters
    return {"events": {"__df__": {"val": val}}}
''',
    ),
    dict(
        id="s17_nested_totals",
        types=["groupby_numeric_only", "level_reduction"],
        entry="nested_totals",
        legacy='''
def nested_totals(df):
    by_pair = df.groupby(['region', 'store']).sum()
    return by_pair.sum(level=0)
''',
        modern='''
def nested_totals(df):
    by_pair = df.groupby(['region', 'store']).sum(numeric_only=True)
    return by_pair.groupby(level=0).sum()
''',
        gen='''
def gen(rng):
    n = rng.randint(4, 12)
    return {"df": {"__df__": {"region": [rng.choice(["east", "west"]) for _ in range(n)],
                              "store": [rng.randint(1, 4) for _ in range(n)],
                              "sales": [rng.randint(0, 100) for _ in range(n)],
                              "note": [rng.choice(["a", "b"]) for _ in range(n)]}}}
''',
    ),
    dict(
        id="s18_code_ends",
        types=["str_replace_regex", "series_positional_getitem"],
        entry="code_ends",
        legacy='''
def code_ends(s):
    cleaned = s.str.replace('[^A-Z]', '')
    return [cleaned[0], cleaned[-1]]
''',
        modern='''
def code_ends(s):
    cleaned = s.str.replace('[^A-Z]', '', regex=True)
    return [cleaned.iloc[0], cleaned.iloc[-1]]
''',
        gen='''
def gen(rng):
    n = rng.randint(2, 6)
    vals = [f"{rng.choice('ABC')}{rng.randint(0, 9)}{rng.choice('XYZ')}-{rng.choice('pq')}" for _ in range(n)]
    idx = [f"id{i}" for i in rng.sample(range(100), n)]
    return {"s": {"__series__": {"values": vals, "index": idx, "name": "code"}}}
''',
    ),
    dict(
        id="s19_grade_counts",
        types=["applymap", "groupby_observed"],
        entry="grade_counts",
        legacy='''
def grade_counts(df):
    labels = df.applymap(str.upper)
    cat = pd.Categorical(labels['grade'], categories=['A', 'B', 'C', 'D'])
    return labels.groupby(cat)['grade'].count()
''',
        modern='''
def grade_counts(df):
    labels = df.map(str.upper)
    cat = pd.Categorical(labels['grade'], categories=['A', 'B', 'C', 'D'])
    return labels.groupby(cat, observed=False)['grade'].count()
''',
        gen='''
def gen(rng):
    n = rng.randint(2, 8)
    return {"df": {"__df__": {"grade": [rng.choice("abc") for _ in range(n)],
                              "dept": [rng.choice(["ops", "eng"]) for _ in range(n)]}}}
''',
    ),
    dict(
        id="s20_return_stats",
        types=["np_product_family", "np_nan_inf_aliases", "ndarray_ptp"],
        entry="return_stats",
        legacy='''
def return_stats(prices):
    p = np.asarray(prices, dtype=float)
    rets = p[1:] / p[:-1]
    rets[rets > 3] = np.NaN
    clean = rets[~np.isnan(rets)]
    return [np.product(clean), np.cumproduct(clean)[-1], clean.ptp(), np.sometrue(np.isnan(rets))]
''',
        modern='''
def return_stats(prices):
    p = np.asarray(prices, dtype=float)
    rets = p[1:] / p[:-1]
    rets[rets > 3] = np.nan
    clean = rets[~np.isnan(rets)]
    return [np.prod(clean), np.cumprod(clean)[-1], np.ptp(clean), np.any(np.isnan(rets))]
''',
        gen='''
def gen(rng):
    prices = [round(rng.uniform(10, 50), 2) for _ in range(rng.randint(3, 8))]
    if rng.random() < 0.4:
        i = rng.randrange(1, len(prices))
        prices[i] = round(prices[i - 1] * 4, 2)
    return {"prices": prices}
''',
    ),
]
