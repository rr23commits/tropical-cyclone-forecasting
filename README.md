# Tropical Cyclone Track & Intensity Forecasting

A reproducible, track-only research project for forecasting North Atlantic tropical-cyclone position and maximum sustained wind from historical IBTrACS observations. It compares simple motion baselines with statistical and recurrent models on a chronological held-out storm set, and includes a local historical-replay interface for inspecting archived predictions.

## Research question

How much can short-range tropical-cyclone track and intensity forecasting improve over persistence and constant-motion baselines when using only recent historical storm-track observations?

## Current scope

The current experiment uses North Atlantic IBTrACS v04r01 records aligned to HURDAT `USA_*` fields. It operates on exact six-hour observations and uses five prior states (24 hours) of latitude, longitude, and wind, together with derived history features. Each prediction origin targets direct forecasts at +6, +12, +24, and +48 hours for:

- Position, represented as latitude/longitude displacement and evaluated as great-circle track error.
- Maximum sustained wind, represented as change from the issue-time wind.

The split is chronological and storm-level: training covers 1980–2015, validation covers 2016–2019, and the held-out test period covers 2020–2025. A storm belongs to exactly one split.

This is a track-only experiment. Environmental reanalysis and satellite imagery are not part of the final model in this repository.

## Final experimental protocol

The source is IBTrACS v04r01, filtered to North Atlantic `TRACK_TYPE=main` records with six-hourly, complete HURDAT-derived `USA_LAT`, `USA_LON`, and `USA_WIND` states and `USA_AGENCY=hurdat_atl`. Each direct forecast uses exactly five observations, `t-24`, `t-18`, `t-12`, `t-6`, and `t`; future track is represented as local north/east displacement and intensity as wind change. Predictions are reconstructed to latitude/longitude and scored against the target state with great-circle Haversine distance in kilometres; wind errors are reported in knots.

Storms are assigned once from their first season: train through 2015, validation 2016--2019, and held-out test from 2020 onward. Thus no storm can contribute rows to more than one split. Ridge standardization and fitting use training rows only; Ridge alpha and GRU width/early stopping use validation only. The compact GRU uses seed 42 and deterministic Torch algorithms. Every model is scored on the same eligible held-out origins at each horizon.

## Models

- **Persistence:** holds the current position and wind.
- **Constant Motion:** extrapolates recent motion; wind remains persistent.
- **Ridge:** a multi-output ridge-regression baseline using the historical-track features.
- **GRU:** a compact recurrent model using the same historical-track inputs.

## Held-out results

The table below summarizes the archived 2020–2025 test results. Each cell is **mean track error in km / wind MAE in kt**; lower is better.

| Horizon | Persistence | Constant Motion | Ridge | GRU |
| --- | ---: | ---: | ---: | ---: |
| +6h | 123.5 / 4.4 | 28.3 / 4.4 | 26.4 / 4.0 | 27.7 / 3.8 |
| +12h | 241.5 / 8.3 | 69.5 / 8.3 | 62.4 / 7.4 | 64.0 / 6.8 |
| +24h | 462.8 / 14.2 | 179.4 / 14.2 | 153.5 / 12.8 | 152.8 / 11.4 |
| +48h | 860.5 / 20.7 | 463.3 / 20.7 | 363.5 / 18.3 | 361.4 / 16.4 |

The complete evaluation includes median and P95 track error, wind RMSE and bias, storm-level bootstrap confidence intervals, directional residual diagnostics, and a fixed three-seed GRU robustness table in `results/track_only_evaluation/`. At +48h, GRU mean track error is 361.4 km and its P95 is 792.1 km, illustrating the long error tail at extended lead times.

These headline means are origin-weighted. Since origins from the same storm are correlated, the report also supplies storm-macro means and 95% bootstrap intervals. The paired GRU--Ridge table reports both its point differences and intervals as equal-storm-weighted paired estimates on identical origins. It supports Ridge's lower +6h mean track error; later track-error intervals include zero. GRU has lower wind MAE at +12h, +24h, and +48h under the paired storm bootstrap; the +6h wind interval includes zero. These are comparisons within this fixed retrospective experiment, not operational-skill claims.

## Error analysis

The archived analysis preserves both favorable and unfavorable outcomes:

- Track error grows substantially with lead time for every model; learned models improve materially on the two simple baselines at +24h and +48h.
- Ridge has the lower mean track error at +6h and +12h, while GRU is slightly lower at +24h and +48h. A paired, storm-level GRU-versus-Ridge comparison is included for identical held-out origins rather than treating overlapping window origins as independent storms.
- GRU has lower wind MAE than Ridge at all four horizons, but wind errors increase with lead time.
- Both learned models show intensity-change difficulty: forecasts tend to underpredict strengthening and overpredict weakening, especially at longer horizons.
- The analysis reports performance by target motion, issue intensity, realized wind change, and test year (2020–2025). Bootstrap intervals resample storms, preserving within-storm correlation among overlapping forecast origins.

Generated tables and plots are written to `results/track_only_evaluation/`; the source error analysis is retained in `results/track_only_error_analysis/`.

## Research replay

The frontend is a map-first local workstation for replaying historical IBTrACS timestamps, observed tracks, and archived model predictions. It supports storm selection, continuous timeline scrubbing, model comparison, available forecast leads, and intensity/error views.

**`RESEARCH REPLAY // NON-OPERATIONAL`** — it is an archive viewer for historical held-out results, not a live or operational forecasting system.

## Repository layout

```text
src/                         Data preparation, models, evaluation, and replay export
models/                      Saved model artifacts when produced locally
results/                     Archived experiment outputs and evaluation reports
frontend/                    Static historical-replay interface
tests/                       Unit and integration-style checks
research.md                  Experiment protocol and research notes
Makefile                     Common test, report, and local-serving commands
```

Key modules include `src/ibtracs.py` for data windows and split construction, `src/baselines.py` and `src/gru.py` for the models, `src/evaluation.py` for metrics, and `src/report_evaluation.py` for the consolidated report.

## Setup and run

Create an environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw IBTrACS CSV is intentionally not committed. Obtain the North Atlantic v04r01 file locally, then provide its path when reproducing model runs or exporting replay data:

```bash
export IBTRACS_CSV=/path/to/ibtracs.NA.list.v04r01.csv
python3 -m src.run_baselines "$IBTRACS_CSV"
python3 -m src.run_gru "$IBTRACS_CSV"
python3 -m src.run_error_analysis "$IBTRACS_CSV"
make evaluation-report
IBTRACS_CSV="$IBTRACS_CSV" make eda gru-seed-robustness provenance
make frontend-data
make serve
```

Generate the consolidated evaluation report from the archived error outputs:

```bash
make evaluation-report
```

Run the test suite:

```bash
make test
```

To regenerate every final track-only output from the source CSV, including the three fixed GRU seeds, run:

```bash
IBTRACS_CSV=/path/to/ibtracs.NA.list.v04r01.csv make reproduce-track-only
```

Serve the frontend through the standard local workflow:

```bash
make serve
```

`make serve` regenerates the replay payload and serves `frontend/` at `http://localhost:8000`. If the IBTrACS file is stored outside the exporter's local default path, generate an identity-enriched payload first and serve the static directory directly:

```bash
python3 -m src.export_replay --ibtracs "$IBTRACS_CSV"
python3 -m http.server 8000 --directory frontend
```

Generated replay data lives under `frontend/data/` and is excluded from version control.
`results/README.md` separates final reportable artifacts from retained historical phases and incomplete Genesis feasibility outputs.

## Limitations and next step

These results are limited to historical-track inputs, a North Atlantic subset, four deterministic baselines, and fixed six-hour forecast origins. They do not establish performance for operational forecasting, real-time data availability, other basins, or satellite-informed models.

The separate Genesis ERA5 extension is incomplete: 899 of 2,792 immutable requests are validated, but only 374 candidate histories from 207 storms have complete pressure/SST coverage, with 370 training candidates, four validation candidates, and no test candidates. It is not used for feature extraction, modelling, or any reported predictive result. Acquisition and the raw journal remain outside this final experiment.

The planned next research step is a separately scoped TCIR satellite ablation: first establish reliable track-to-image alignment and a reproducible image-only contribution, then compare it against this track-only reference without changing the held-out evaluation protocol.
