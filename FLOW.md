# Execution Flow

## Phase 1 data flow

`src/audit_phase1.py` calls `load_ibtracs()` in `src/ibtracs.py`, then `filter_north_atlantic()`, then `build_forecast_samples()` for each horizon.

The loader skips the IBTrACS units row and parses timestamps/numeric USA fields. Filtering keeps the audited North Atlantic HURDAT states. Sample construction groups rows by `SID`, requires five contiguous six-hour historical states plus a valid future state, derives history-only motion features, and returns direct displacement/wind-change targets with metadata.

`storm_split_map()` assigns each `SID` atomically to train, validation, or test after split boundaries are selected.

## Phase 2 baseline flow

`python3 -m src.run_baselines <csv>` loads the Phase 1 sample sets, assigns whole-storm chronological splits, fits/selects ridge only on training and validation rows, and scores every baseline on the identical test rows. `src/evaluation.py` reconstructs predicted coordinates before applying Haversine track error and wind MAE/RMSE. The final run fixes history to five states (24 hours), evaluates +6/+12/+24/+48-hour targets, and writes `results/track_only_baselines.csv`.

## Phase 3 GRU flow

`python3 -m src.run_gru <csv>` uses the same filtered samples and split map as Phase 2. For each horizon, `src/gru.py` fits normalization statistics on training rows, trains width-16 and width-32 GRU candidates, selects the best validation checkpoint, and predicts the unchanged test rows. The final runner evaluates GRU and all three baselines through `src/evaluation.py`, writes `results/track_only_gru_comparison.csv`, and `src.plot_track_results` renders its track-error SVG without another dependency.

## Held-out error-analysis flow

`python3 -m src.run_error_analysis <csv>` recreates exactly the final split and selected-model procedure, but changes no settings and writes no model state. `src.error_analysis.error_rows()` combines each test origin’s source metadata, reconstructed prediction, track error, wind error, and descriptive intensity/motion/wind-change categories. `summarize()` writes aggregate and conditional tables, with storm-clustered bootstrap intervals; it never treats serial windows as independent storms.

## Static research-replay frontend flow

`make frontend-data` calls `src.export_replay.build_payload()` on existing `results/track_only_error_analysis/held_out_errors.csv` and `overall.csv`, with the locally available official IBTrACS CSV only to map each SID to its real name, basin, and ATCF ID. The exporter groups repeated per-model rows into one observed IBTrACS track per SID and one forecast set per issue time/horizon, retaining target states, predictions, and archived errors. `frontend/app.js` loads that payload and the bundled local country-boundary topology, renders an SVG geographic map with pan/zoom/fit, and binds every observed timestamp to slider, scrub, click, and playback state. `togglePanel()` independently changes the collapsed panel class and the corresponding map workspace margin, then redraws after the layout transition. `make serve` exposes the static directory only; no model execution occurs in the browser.

## Phase 4 history-ablation flow

`python3 -m src.run_history_ablation <csv>` builds 2/3/5/7/9-state samples through the existing Phase 1 builder. Each length keeps its own eligible training windows, while `align_common_origins()` limits validation and test evaluation to shared `(SID, issue_time, target_time)` keys. The GRU selection procedure remains unchanged; the runner chooses the lowest common-validation track error per horizon and writes `results/phase4_history_ablation.csv`.

## ERA5 acquisition and extraction flow

This is retained historical implementation only. ERA5/CDS is frozen and is not part of the current experiment or execution plan.

`python3 -m src.acquire_era5 <csv> --concurrency 4` rebuilds the selected-history origin union using `(horizon, history_steps) = (6,9), (24,7), (48,9)`, deduplicates it by `(SID, issue_time)`, then retains September--October dates in `request_manifest_sep_oct.csv`. It groups those dates into 264 monthly pressure-wind, humidity, and SST CDS requests. The repository-canonical `data/raw/era5/v1` directory owns the process lock and atomically persisted journal even when `--raw-dir` changes output storage. The lock-owning coordinator validates cached targets, journals canonical payload/fingerprint/dataset before `retrieve()`, and durably records each returned datastore job ID. It queues at most four independent recorded jobs at once; each worker has a private CDS client and job-specific `.part` file, while only the coordinator atomically saves returned job state. If submission returns an ambiguous error without an ID, resume paginates account jobs and compares each server request to the journaled payload; exactly one match is journaled and joins the queue, while zero or multiple matches abort without `retrieve()`. A successful job obtains `Remote.get_results().location` and `content_length`; `curl --continue-at -` writes the same job-specific `.part` file, with three retries and 900-second transfer/retry limits. The parent checks byte count, validates, and atomically promotes only a successful attempt. Exhaustion records `download_error` without replacing the job; independent queued jobs finish, then the CLI exits non-zero before validation or feature extraction. `extract_features()` receives the same September--October origin subset, so cached off-season files cannot enter `results/phase5_era5_features.csv`. A failed validation or CDS job aborts before feature extraction; no missing value is imputed.
