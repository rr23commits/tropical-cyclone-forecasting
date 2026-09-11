# Handover

## Current state

This is a B.Tech tropical-cyclone track and wind forecasting project. The primary, end-to-end result is now the reproducible IBTrACS track-only experiment: 24-hour (five-state) history at six-hour cadence, direct +6/+12/+24/+48-hour track-displacement and wind forecasts, chronological whole-storm splits, baselines, and a compact GRU. ERA5/CDS is frozen: do not change, resume, or use it for this project phase. Satellite data is not implemented.

The Phase 4-derived union has 2,472 issue dates and 10,867 distinct `(SID, issue_time)` origins. The prior ERA5 September--October plan and all of its state remain preserved as historical work, but it is explicitly out of scope for the current deliverable.

The original July pressure-wind job `e6585a2b-5393-43a2-b553-b29507b0af8d` was resumed without resubmission and is locally validated. A one-time operator recovery archived the provenance-less legacy 502 record for `sst_198511.nc`, then submitted the current manifest-derived request once as CDS job `a1d556f5-df45-460e-a592-86a21eb44f13`; its 3,173,019-byte result validated and was atomically promoted. The September 1981 pressure-wind job `ad689dd9-00e7-4e14-a4ad-fcf187331f38` was recovered by the curl path: it resumed from byte 90,238,720, reached its CDS-reported 137,340,043 bytes, validated its 100 manifest timestamps and 0.25° grid, and atomically promoted without resubmission. The June 1981 pressure-wind file was manually recovered but its journal entry remains `submitted` until the pipeline validates it. `src.acquire_era5` holds an exclusive process lock, journals every request ID, and, once a recorded job is successful, gets its CDS results asset URL and expected byte count then runs system `curl` against the same job-specific `.part` file. Curl resumes with `--continue-at -`, retries up to three times, and has 900-second per-transfer and retry-window limits. The parent requires byte-count verification, NetCDF validation, then atomic promotion. Curl failure is journaled as `download_error` and exits non-zero; no replacement job is submitted. `Results.download()` and urllib are not used for transfers. The existing June 1981 pressure-wind job `a20b40c6-ab81-495b-b08c-7a6fed63afc1` remains recovery-only; no feature table exists. The durable journal is `data/raw/era5/v1/request_jobs.json` (ignored by Git).

## Working and verified

- Full tests: `KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest discover -s tests -p 'test_*.py' -v` (46 tests passed after the error-analysis addition). These include mocked ERA5 tests only; no CDS request is made.
- Run the Phase 1 audit: `python3 src/audit_phase1.py /path/to/ibtracs.NA.list.v04r01.csv`.
- Run baselines: `python3 -m src.run_baselines /path/to/ibtracs.NA.list.v04r01.csv`.
- Run the final baselines: `KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_baselines /path/to/ibtracs.NA.list.v04r01.csv --output results/track_only_baselines.csv`.
- Run the final GRU comparison: `KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_gru /path/to/ibtracs.NA.list.v04r01.csv --output results/track_only_gru_comparison.csv`.
- Render the final chart: `python3 -m src.plot_track_results results/track_only_gru_comparison.csv --output results/track_only_track_error.svg`.
- Recreate held-out error analysis: `KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_error_analysis /path/to/ibtracs.NA.list.v04r01.csv --output-dir results/track_only_error_analysis`.
- Run history ablation: `python3 -m src.run_history_ablation /path/to/ibtracs.NA.list.v04r01.csv`.
- The raw audit CSV is outside Git at `/private/tmp/ibtracs.NA.list.v04r01.csv` in this environment.

## Important constraints

- Keep only `USA_*` values with `USA_AGENCY == hurdat_atl`; do not mix wind conventions.
- Use exact six-hour states and whole-storm chronological splits; never random row splits.
- Keep raw data out of Git. Do not add environmental or satellite paths without a separately scoped next phase.
- The checked-in virtual environment lacks Torch; use the system `python3` used for the recorded final result.

## Next

Use the final artifacts in `results/track_only_baselines.csv`, `results/track_only_gru_comparison.csv`, `results/track_only_track_error.svg`, and `results/track_only_error_analysis/` for the report/viva. The error analysis exports all held-out model predictions plus overall, intensity, motion, wind-change, year, and per-storm summaries. It uses 2,000 seed-42 storm-cluster bootstrap resamples; do not report origin-level uncertainty as independent. The optional TropiCycloneNet environmental experiment is not feasible as a quick extension: no compatible local subset exists, and its documented storm-centered fields/labels require a new alignment study. Do not download or adapt it in this project phase. See `FEATURES/FEATURE-006-track-only-final-experiment.md`.
