# Task dataset: how it was built and its known limits

**Frozen splits** (`tasks/splits/manifest.json`, hashes there): train 1,762 · heldout_id 200 · heldout_cat 138 (60 `mad`, 52 `lookup`, ~26 `weekofyear`).

## Pipeline

Gemma 4 31B (FP8, one H100, vLLM) wrote ~4,550 candidate tasks: 3,250 training-type and 1,300 held-out-type. Every candidate goes through `tasks/filter_tasks.py`, which trusts nothing the generator claims. Each kept task's legacy code runs on 12 inputs, breaks or changes behavior in modern on at least 80% of them, and has a migration that scores 1 under the full reward. Its types are re-derived from the code and the runtime messages. Comments are stripped because the generator annotated deprecated lines. Near-duplicates are clustered by token-shingle Jaccard similarity (≥ 0.7). Splits are assigned by cluster so near-twins never straddle train and test, and train keeps at most 5 tasks per cluster.

## Findings

- **Generator repetitiveness.** For narrow single-API requests, 34–69% of responses were exact structural duplicates. Raising the temperature to 1.3 did not help. 133 accepted `weekofyear` tasks collapsed to ~26 distinct ones. Per-type held-out results therefore carry wide confidence intervals; the pooled held-out-category pass rate is the headline number.
- **Selection bias toward the generator's competence.** A task is kept only if the generator's own migration is correct. Hard variants the generator gets wrong are filtered out even when the task itself is valid. Measured case: append loops that start from an empty DataFrame produce float64 columns in pandas 1.3. The natural list-of-dicts migration gives int64 and is (correctly) rejected, so these dtype-preservation tasks are under-represented. 46 of 273 migration rejections were dtype-only. Possible fix: a repair round that feeds the generator the output diff.
- **The textbook replacement for `sum(level=0)` is not equivalent.** `groupby(level=0).sum()` sorts groups, while `sum(level=)` kept first-appearance order. The equivalent form is `groupby(level=0, sort=False).sum()`. The original verification probe and a seed task used the textbook form and passed only because their inputs were sorted. Both now use unsorted inputs. Same lesson as elsewhere: a check is only as strong as the inputs it exercises.
- **Comparator validated against false negatives.** Of 273 migration rejections, 56 differed only in dtype, names, index, or type. A manual review of 15 rejections (8 of them in those categories) found no case where a correct migration was rejected.
- **An uncatalogued deprecation.** pandas 2.2's "Downcasting object dtype arrays on .fillna" FutureWarning appeared in 22 candidates. These are rejected as unknown breakage; it is a candidate type for a later version.
