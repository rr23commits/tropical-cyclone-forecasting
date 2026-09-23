# Final Implementation

## 1. Data loading and preprocessing

Download the official NOAA IBTrACS v04r01 North Atlantic basin CSV outside Git (or to an ignored raw-data directory). Load it with the units row skipped; preserve `NA` as a basin value rather than parsing it as missing. Parse `ISO_TIME` as UTC and numerical USA fields as numbers.

Retain `BASIN == NA`, `TRACK_TYPE == main`, `SEASON` 1980–2025, and `USA_AGENCY == hurdat_atl`. Keep only the exact six-hour UTC ticks (00, 06, 12, 18) with valid `USA_LAT`, `USA_LON`, and `USA_WIND`, and eligible status values `TD`, `TS`, `HU`, `TY`, `ST`, `TC`, `SD`, or `SS`. Use `SID` as the storm key. Reject duplicate `(SID, ISO_TIME)` states and do not invent values for missing observations; irregular non-six-hour records are not input states.

Keep `USA_PRES` separate from the core pipeline. Its complete-case subset can support a later pressure experiment without reducing the track-and-wind dataset.

## 2. Splits and samples

Assign each entire storm to a chronological train, validation, or test period using its first season; no `SID` may span splits. The fixed boundaries are 1980--2015 for training, 2016--2019 for validation, and 2020--2025 for held-out testing. Every model uses this same deterministic split map.

For an issue time `t`, require a continuous five-state window at `t-24`, `t-18`, `t-12`, `t-6`, and `t`, plus a valid state at each requested target time. Build direct examples separately for +6, +12, +24, and +48 hours. A missing required six-hour state makes that particular example ineligible; it does not discard the whole storm.

## 3. Inputs, features, and targets

The initial input sequence contains `USA_LAT`, `USA_LON`, and `USA_WIND` for the five states. Derive only historical quantities: consecutive geographic displacements, translation speed/direction, and wind changes. Handle longitude wraparound when deriving motion; retain geographic coordinates for final reconstruction and scoring.

For every horizon, predict north/east displacement from the position at `t` and wind change from wind at `t`. Convert predicted displacement back to latitude/longitude and predicted wind back to physical units before evaluation. Fit any scaling, encoding, or imputation rule on training data only, then apply it unchanged to validation and test data.

## 4. Baselines

- **Persistence:** predict the state at `t` for every horizon.
- **Constant motion:** extrapolate the latest six-hour displacement to the target horizon; retain current wind as the default intensity forecast.
- **Regularized lagged regression:** flatten the history and its derived features, then fit regularized regression models for displacement and wind change. Train direct models per horizon.

These baselines must use exactly the same eligible cases and reconstructed outputs as the neural model.

## 5. First neural experiment

The final neural reference is a compact one-layer GRU consuming the five-step feature sequence and producing direct displacement and wind-change outputs. Train direct models per horizon; select width and stopping point using validation data only. The architecture is frozen for the reported experiment.

## 6. Evaluation and leakage checks

For each split and horizon, report great-circle track error in kilometres plus wind MAE and RMSE. Report the number of forecast origins and distinct storms, and compare all models on the same horizon-specific test cases. Summarize mean error and, where useful, median or storm-level error so a few long-lived storms do not obscure results.

Prevent leakage by enforcing storm-level chronological splits, using only observations at or before `t`, fitting preprocessing on training data alone, and never using a future observed position to derive an issue-time feature. Do not recursively use true future states while scoring multi-horizon forecasts.

## 7. Experiment records and reproducibility

The archived seed-42 primary result is complemented by fixed seeds 42, 43, and 44, each selected solely on validation data and reported without choosing a test winner. `src.write_provenance` writes the IBTrACS SHA-256, code revision, Python/package versions, frozen split/configuration, seeds, and output hashes to `results/track_only_provenance.json`. `src.report_eda` writes data eligibility/missingness, training-only distributions, split/horizon counts, and deterministic training-storm illustrations to `results/track_only_eda/`. Keep raw data out of Git; commit only code, configuration, and lightweight result summaries.

## 8. Deferred extensions

ERA5/CDS is frozen and satellite/TCIR is not implemented. No environmental or imagery result is part of the completed experiment; they require a separately scoped future study using the unchanged leakage safeguards and storm-wise split policy.
