# Results index

## Final reportable track-only artifacts

## Separate monthly activity artifacts

`monthly_activity_ts/` is a separate monthly storm-count experiment. It must not be mixed with `track_only_*` tables: it counts retained SIDs by first-observation month and evaluates monthly one-step forecasts for 2016–2025.

Its `xgboost_*` files are the isolated XGBoost extension; the unprefixed classical monthly files are unchanged.

Its `lstm_*` files are the isolated small-LSTM extension; they likewise do not alter classical or XGBoost outputs.

Its `prophet_*` files are the isolated Prophet extension with yearly seasonality and no external regressors.

Its `ensemble_*` files are the final XGBoost–Prophet validation-weighted blend; component artifacts are not rewritten.

The separate presentation page is `frontend/monthly.html`; regenerate its derived `frontend/data/monthly.json` with `make monthly-frontend-data`. It is a read-only projection of these artifacts and is not part of model execution.

- `track_only_baselines.csv`, `track_only_gru_comparison.csv`: frozen single-seed-42 primary comparison.
- `track_only_gru_seed_results.csv`, `track_only_gru_seed_summary.csv`: fixed seeds 42, 43, and 44; all runs are reported, with no test-set selection.
- `track_only_error_analysis/`: held-out row-level errors and descriptive slices.
- `track_only_evaluation/`: final aggregate, paired storm-macro comparison, residual tables, and figures.
- `track_only_eda/`: data-quality, split/count, training-only distribution, and representative-training-storm figures.
- `track_only_provenance.json`: exact source hash, code revision, environment, frozen protocol, and output hashes.

## Historical or separate-scope artifacts

`phase2_baselines.csv`, `phase3_gru_comparison.csv`, `phase4_history_ablation.csv`, and `track_only_track_error.svg` are retained historical phase outputs; they are not the final report table. `genesis_*`, `gridsat_s3_index/`, and `lps_tcc_crosswalk/` are feasibility/acquisition artifacts for the incomplete extension and are not predictive results.
