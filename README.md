# Tropical Cyclone Forecasting

Reproducible research code for two separate North Atlantic experiments based on historical IBTrACS observations:

1. **Track-only forecasting:** direct 6-, 12-, 24-, and 48-hour position and maximum-wind forecasts for already identified storms.
2. **Monthly storm activity:** one-step-ahead forecasts of the monthly count of retained storms from 1980–2025.

The experiments use separate code paths and result namespaces. The monthly count experiment is not a replacement for, or an aggregation of, the track-forecasting evaluation.

## Data

The source is the official IBTrACS v04r01 North Atlantic CSV, using the HURDAT-aligned `USA_*` fields. The pipelines retain records with:

- `BASIN=NA`
- `TRACK_TYPE=main`
- `USA_AGENCY=hurdat_atl`
- seasons 1980–2025
- exact six-hour UTC timestamps
- valid latitude, longitude, and maximum sustained wind

The raw CSV is intentionally kept outside Git. Provide its local path when running the experiment commands.

## Experiment 1: track-only forecasting

### Question

How much can short-range track and intensity forecasting improve over persistence and constant-motion baselines when using only recent historical storm-track observations?

### Protocol

Each forecast origin uses the five observations at `t-24`, `t-18`, `t-12`, `t-6`, and `t`. Models predict direct north/east displacement and wind change for +6, +12, +24, and +48 hours. Position predictions are reconstructed and scored with great-circle distance; wind errors are reported in knots.

Storms are assigned to one chronological split using their first season:

- train: 1980–2015
- validation: 2016–2019
- test: 2020–2025

No storm contributes forecast origins to more than one split. Preprocessing and model selection use training and validation data only.

### Models

- **Persistence:** holds the current position and wind.
- **Constant motion:** extrapolates recent motion; wind remains persistent.
- **Ridge:** multi-output ridge regression over historical-track features.
- **GRU:** one-layer recurrent model using the same history inputs.

### Held-out test results

Values are mean track error in kilometres / wind MAE in knots. Lower is better.

| Lead | Persistence | Constant motion | Ridge | GRU |
|---|---:|---:|---:|---:|
| +6 h | 123.5 / 4.4 | 28.3 / 4.4 | 26.4 / 4.0 | 27.7 / 3.8 |
| +12 h | 241.5 / 8.3 | 69.5 / 8.3 | 62.4 / 7.4 | 64.0 / 6.8 |
| +24 h | 462.8 / 14.2 | 179.4 / 14.2 | 153.5 / 12.8 | 152.8 / 11.4 |
| +48 h | 860.5 / 20.7 | 463.3 / 20.7 | 363.5 / 18.3 | 361.4 / 16.4 |

At +48 hours, the GRU mean track error is 361.4 km and its archived P95 track error is 792.1 km. The complete report also contains medians, P95 values, wind RMSE and bias, storm-level bootstrap intervals, paired GRU–Ridge comparisons, and error slices by year, motion, intensity, and realized wind change. These are retrospective results on this fixed dataset, not operational-skill claims.

## Experiment 2: monthly storm activity

### Question

Can the historical seasonal and temporal structure of retained North Atlantic storm activity support one-step-ahead monthly count forecasts under a strictly chronological evaluation?

### Target and audit

The target is the number of **retained SIDs whose first retained six-hour observation falls in each month**. Each SID is counted once. This is a monthly retained-storm activity series; it is **not a physical tropical-cyclone genesis count** and should not be interpreted as one.

The generated audit contains 552 monthly observations from January 1980 through December 2025, 738 retained SIDs, and 304 observed zero-count months. Zeros are real observations, not missing values.

The fixed splits are:

- train: 1980–2015, 432 months
- validation: 2016–2019, 48 months
- test: 2020–2025, 72 months

Every model uses the same expanding-window, one-step-ahead validation and test timestamps. Settings are selected on validation only and frozen before test evaluation.

### Models and selected settings

- **Seasonal naive:** lag-12 (`t-12`).
- **ARIMA:** selected order `(0, 1, 0)`.
- **SARIMA:** `(0, 0, 0) × (0, 1, 0, 12)`.
- **Additive Holt–Winters:** fitted settings `(0.2, 0.2, 0.3)` as recorded in the audit.
- **XGBoost:** lags 1, 2, 3, 6, and 12; month sine/cosine; past-only trailing mean; 100 estimators, depth 1, learning rate 0.05, trailing mean enabled.
- **LSTM:** one layer, input size 1, hidden size 8, output size 1; 12-month input window; learning rate 0.01, 30 epochs, seed 42.
- **Prophet:** additive yearly seasonality, Fourier order 5, changepoint prior scale 0.1, seasonality prior scale 10.0, and no external regressors.
- **XGBoost–Prophet ensemble:** validation-only XGBoost weight 0.55 and Prophet weight 0.45.

### Test results

| Model | MAE | RMSE | MASE |
|---|---:|---:|---:|
| Seasonal naive | 1.181 | 2.017 | 1.083 |
| ARIMA | 1.528 | 2.392 | 1.401 |
| SARIMA | 1.181 | 2.017 | 1.083 |
| Additive Holt–Winters | 2.093 | 2.606 | 1.920 |
| XGBoost | 1.010 | 1.506 | 0.926 |
| LSTM | 1.254 | 1.947 | 1.150 |
| Prophet | 1.030 | 1.431 | 0.945 |
| XGBoost–Prophet ensemble | 1.016 | 1.461 | 0.931 |

Metrics are monthly-count errors over the 72-month test period. The ensemble weight was selected using validation forecasts only; the test period was not used for model or weight selection.

## Interactive frontend

The static frontend has two linked views:

- `frontend/index.html`: historical Spatial Trajectory replay of archived storm forecasts. It is labelled non-operational and does not run live inference.
- `frontend/monthly.html`: Monthly Activity research presentation with the full count series, a test-period-focused forecast view, month-of-year seasonality, decomposition, model comparison, and interactive hover/crosshair tooltips.

The Monthly Activity page reads `frontend/data/monthly.json`, which is generated from the existing monthly experiment artifacts. The Spatial Trajectory map and the Monthly Activity page do not share model results or evaluation logic.

## Repository layout

```text
src/                         Data loading, models, evaluation, and exporters
tests/                       Unit and integration-style tests
frontend/                    Static Spatial Trajectory and Monthly Activity pages
results/                     Archived experiment outputs and plots
requirements.txt             Python dependencies
Makefile                     Reproducibility and local-serving commands
```

Important modules include `src/ibtracs.py` for filtering and chronological samples, `src/baselines.py` and `src/gru.py` for track models, `src/evaluation.py` for track metrics, `src/monthly_activity.py` for the classical monthly experiment, and the `src/monthly_activity_*` modules for the monthly extensions.

## Reproduce the experiments

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the path to the local IBTrACS CSV:

```bash
export IBTRACS_CSV=/path/to/ibtracs.NA.list.v04r01.csv
```

Run the test suite:

```bash
make test
```

Reproduce the track-only outputs:

```bash
IBTRACS_CSV="$IBTRACS_CSV" make reproduce-track-only
```

Run the monthly experiment in order:

```bash
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity-xgboost
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity-lstm
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity-prophet
make monthly-activity-ensemble
make monthly-frontend-data
```

Generate the consolidated track evaluation report:

```bash
make evaluation-report
```

Regenerate the replay payload with the local CSV, regenerate the monthly frontend payload, and serve both pages:

```bash
python3 -m src.export_replay --ibtracs "$IBTRACS_CSV"
make monthly-frontend-data
python3 -m http.server 8000 --directory frontend
```

Then open `http://localhost:8000/`. The raw input remains external to Git; generated frontend payloads and experiment outputs are derived artifacts.

## Limitations

- The track experiment uses historical track and intensity observations only. It does not establish performance for real-time operations, other basins, environmental predictors, or satellite imagery.
- The monthly target counts retained storm IDs by first retained observation month. It is not a genesis label, and the 738 retained SIDs are defined by the documented filtering rules.
- The monthly dataset has only 552 observations and 304 zero months. Results should be read as a bounded retrospective comparison, not as a general claim about future climatology.
- The monthly models use chronological expanding forecasts, but the test period remains a single 72-month historical window.
- The frontend is a static archive viewer and research presentation. It performs no live inference.
- A separate Genesis/ERA5 feasibility path exists in the repository history, but it is incomplete and is not used by either reported experiment.
