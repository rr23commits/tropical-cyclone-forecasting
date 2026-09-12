# Feature 006: Track-only final experiment

## Goal and scope

Produce the project’s complete, defensible baseline result from local IBTrACS data before adding environmental or satellite data. ERA5/CDS is frozen and TCIR is deferred.

## Flow

`src.run_baselines` and `src.run_gru` load the existing North Atlantic HURDAT-filtered IBTrACS states through `src.ibtracs`, construct five contiguous six-hour states ending at each issue time, split whole storms by first season, then score the held-out test storms through `src.evaluation`.

## Implementation

The existing direct displacement/wind-change builder now accepts the required +12-hour lead in addition to +6/+24/+48 hours. The runners use the same fixed 24-hour history and write sample/storm split counts with their test metrics. `src.plot_track_results` is a small standard-library SVG renderer for the GRU comparison.

## Split and verification

Storms whose first season is 1980--2015 train, 2016--2019 validate, and 2020--2025 test. No storm appears in more than one split and no random row split occurs. `KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest discover -s tests -p 'test_*.py' -v` passed 46 tests after the error-analysis addition.

## Results

The reproducible artifacts are `results/track_only_baselines.csv`, `results/track_only_gru_comparison.csv`, and `results/track_only_track_error.svg`. The GRU improves held-out wind MAE over the ridge baseline at every lead, while ridge remains marginally lower for +6-hour track error.

## Error analysis

`src.run_error_analysis` recreates the frozen experiment and exports `held_out_errors.csv` plus overall, conditional, annual, and per-storm CSVs under `results/track_only_error_analysis/`. The retained categories are issue intensity, realised target motion, and realised wind change; they are descriptive slices, not extra model inputs or causal attribution. Intervals are 95% percentile intervals from 2,000 fixed-seed resamples of whole storms. The +48-hour GRU has a 361.41 km origin-weighted mean track error but a 792.12 km P95, so mean-only reporting would hide material tail risk. Wind forecasts systematically underpredict strengthening and overpredict weakening, especially at longer lead times.

## Archived evaluation report

`make evaluation-report` produces `results/track_only_evaluation/` directly from the held-out predictions already exported by the frozen experiment. `model_horizon_summary.csv` provides mean/median/P95 track error, wind MAE/RMSE/bias, and existing storm-level CIs for all models and leads. `difficulty_summary.csv` and `yearly_stability.csv` retain the established motion, intensity, intensity-change, and 2020--2025 views. `paired_gru_vs_ridge.csv` compares only common origins and reports storm-clustered paired delta intervals. At +6h, GRU's mean track error is 1.34 km higher than Ridge (paired storm CI 0.67 to 2.33 km); at +24/+48h, track differences are uncertain while GRU's wind MAE advantage is retained. The three SVGs visualize model track error, wind MAE, and GRU/Ridge annual track stability.

## Limitations

The test span is six seasons and storm origins within a storm are correlated. The analysis controls the latter through storm-level summaries and resampling, but it is not an independent storm-season replication study. As a track-only model it cannot attribute failures to environmental conditions, shear, SST, or imagery.

## Environmental feasibility

No TropiCycloneNet subset is present locally. Its public description uses a distinct storm-centered grid and different auxiliary/label conventions, so a trustworthy alignment requires a new data study rather than a quick loader. This is intentionally not implemented.
