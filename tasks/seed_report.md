# Seed task check

20/20 tasks pass on 12 inputs each. Training types covered: 24/24.

| task | types | legacy | legacy code in modern | pass |
|---|---|---|---|---|
| s01_group_summary | df_append_loop | OK | FAIL | yes |
| s02_column_totals | series_append, iteritems | OK | FAIL | yes |
| s03_scale_cells | applymap, np_type_aliases | WARN | FAIL | yes |
| s04_group_means_ffill | groupby_numeric_only, fillna_method | OK | FAIL | yes |
| s05_region_totals | level_reduction | WARN | FAIL | yes |
| s06_parse_ids | read_csv_squeeze, str_replace_regex | WARN | FAIL | yes |
| s07_clean_phone | str_replace_regex | WARN | OK | yes |
| s08_sales_by_size | groupby_observed, int64index | OK | FAIL | yes |
| s09_fill_and_ends | chained_inplace, series_positional_getitem | OK | WARN | yes |
| s10_growth_factor | groupby_apply_groups, np_product_family | OK | FAIL | yes |
| s11_capped_ratios | np_type_aliases, np_nan_inf_aliases | WARN | FAIL | yes |
| s12_curve_areas | np_row_stack, np_trapz | OK | WARN | yes |
| s13_selected_spread | np_in1d, ndarray_ptp | OK | FAIL/WARN | yes |
| s14_stack_columns | array_copy_false, find_common_type | OK | FAIL | yes |
| s15_csv_report | to_csv_line_terminator, iteritems | OK | FAIL | yes |
| s16_running_log | df_append_loop, fillna_method | OK | FAIL | yes |
| s17_nested_totals | groupby_numeric_only, level_reduction | WARN | FAIL | yes |
| s18_code_ends | str_replace_regex, series_positional_getitem | WARN | WARN | yes |
| s19_grade_counts | applymap, groupby_observed | OK | WARN | yes |
| s20_return_stats | np_product_family, np_nan_inf_aliases, ndarray_ptp | OK | FAIL | yes |
