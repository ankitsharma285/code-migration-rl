import warnings, pandas as pd, numpy as np
print(f"pandas {pd.__version__}, numpy {np.__version__}")
df = pd.DataFrame({"a": [1]})
checks = [("append", lambda: df.append(df)),
          ("np.float", lambda: np.float(1)),
          ("applymap", lambda: df.applymap(str))]
for name, fn in checks:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            print(f"  {name:9s} {'WARNS: ' + w[0].category.__name__ if w else 'OK'}")
        except Exception as e:
            print(f"  {name:9s} FAILS: {type(e).__name__}")
