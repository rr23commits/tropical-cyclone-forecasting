# Tropical Cyclone Forecasting

Reproducible research code and archived outputs for tropical-cyclone prediction experiments. The project has two deliberately separate research components:

1. **North Atlantic storm-track forecasting and monthly cyclone-activity time series:** retrospective experiments derived from filtered historical IBTrACS observations.
2. **Satellite-based Genesis experiment:** an additional exploratory multi-basin CNN–GRU study using archived GridSat infrared imagery and a frozen Genesis cohort.

The Genesis experiment does not alter the storm-track forecasting protocol, targets, models, or reported results. Its cohort, data path, model, and evaluation are independent.

## Research components

### Storm-track forecasting and held-out replay

This is the primary forecasting study. Given five six-hourly observations at `t−24`, `t−18`, `t−12`, `t−6`, and `t`, models predict direct displacement and wind change at +6, +12, +24, and +48 hours for already identified storms. Predicted positions are reconstructed from north/east displacement and evaluated with great-circle track error; wind errors are in knots.

Storms are assigned atomically by first season:

| Split | Seasons |
|---|---|
| Train | 1980–2015 |
| Validation | 2016–2019 |
| Test | 2020–2025 |

Preprocessing and model selection use training and validation data only. The evaluated models are persistence, constant motion, multi-output ridge regression, and a one-layer GRU.

Held-out test results are mean track error in km / wind MAE in kt (lower is better):

| Lead | Persistence | Constant motion | Ridge | GRU |
|---|---:|---:|---:|---:|
| +6 h | 123.5 / 4.4 | 28.3 / 4.4 | 26.4 / 4.0 | 27.7 / 3.8 |
| +12 h | 241.5 / 8.3 | 69.5 / 8.3 | 62.4 / 7.4 | 64.0 / 6.8 |
| +24 h | 462.8 / 14.2 | 179.4 / 14.2 | 153.5 / 12.8 | 152.8 / 11.4 |
| +48 h | 860.5 / 20.7 | 463.3 / 20.7 | 363.5 / 18.3 | 361.4 / 16.4 |

The GRU's +48 h mean track error is 361.4 km; its archived P95 track error is 792.1 km. Detailed aggregate, paired, residual, and error-slice outputs are under `results/track_only_evaluation/` and `results/track_only_error_analysis/`.

### Monthly cyclone-activity time series

This is a separate chronological time-series experiment, not an aggregation of the storm-track evaluation. The target is the monthly count of retained North Atlantic storm IDs: each retained SID is counted once, in the month of its first retained six-hour observation. It is not a physical genesis count.

The series contains 552 monthly observations from 1980–2025, 738 retained SIDs, and 304 observed zero-count months. All models use expanding-window one-step-ahead forecasts with fixed splits: train 1980–2015, validation 2016–2019, and test 2020–2025.

Models include seasonal naive, ARIMA, SARIMA, additive Holt–Winters, XGBoost with past-only features, a small LSTM, Prophet, and a validation-weighted XGBoost–Prophet ensemble.

| Held-out model | MAE | RMSE | MASE |
|---|---:|---:|---:|
| Seasonal naive | 1.181 | 2.017 | 1.083 |
| ARIMA | 1.528 | 2.392 | 1.401 |
| SARIMA | 1.181 | 2.017 | 1.083 |
| Additive Holt–Winters | 2.093 | 2.606 | 1.920 |
| XGBoost | 1.010 | 1.506 | 0.926 |
| LSTM | 1.254 | 1.947 | 1.150 |
| Prophet | 1.030 | 1.431 | 0.945 |
| XGBoost–Prophet ensemble | 1.016 | 1.461 | 0.931 |

The ensemble weight (XGBoost 0.55, Prophet 0.45) was selected on validation forecasts only. Artifacts for this component are in `results/monthly_activity_ts/`.

### Satellite-based Genesis experiment

The Genesis experiment is an additional exploratory binary-classification study. A CNN encodes each of nine historical 201×201 GridSat infrared images at three-hour spacing; a GRU aggregates the image features and predicts the frozen 24-hour Genesis target, `label_24h`.

The frozen candidate manifest contains 1,276 candidates. After requiring all nine acquired crops, the complete cohort has 1,201 candidates: 750 train, 140 validation, and 311 test. The test split contains 210 positive and 101 negative labels. The cohort spans EP, WP, SI, SP, NI, and SA candidates, plus unresolved-basin candidates. It uses a separate candidate cohort and GridSat crop archive, rather than the North Atlantic storm-track feature windows or monthly-count target.

| Held-out metric | CNN–GRU |
|---|---:|
| Accuracy | 0.611 |
| Precision | 0.683 |
| Recall | 0.790 |
| F1 | 0.733 |
| ROC-AUC | 0.562 |

For context, an always-positive majority-class baseline has accuracy 0.675 and F1 0.806 on this test split; ROC-AUC is undefined for its constant output. The CNN–GRU results are therefore reported as an exploratory imagery result, not as a replacement for the primary storm-track forecasting study.

## Data

The storm-track and monthly components use the official IBTrACS v04r01 North Atlantic CSV and HURDAT-aligned `USA_*` fields. Records are retained when they have `BASIN=NA`, `TRACK_TYPE=main`, `USA_AGENCY=hurdat_atl`, seasons 1980–2025, exact six-hour UTC timestamps, and valid position and maximum sustained-wind values. The raw IBTrACS CSV is not committed; supply its local path through `IBTRACS_CSV`.

The Genesis component uses the frozen manifest at `results/genesis_candidate_manifest.csv`, acquisition-state metadata, and prepared GridSat infrared crops. The public frontend includes the derived Genesis export (`frontend/data/genesis.json`) and its archived WebP frames; the source crop archive is external to Git.

## Frontend

The static frontend provides three linked research views:

- **Spatial Trajectory** (`frontend/index.html`): storm-track forecasting and held-out replay, with archived forecasts and observed verification.
- **Monthly Activity** (`frontend/monthly.html`): the monthly cyclone-activity time series, chronological forecasts, seasonality, decomposition, and held-out model comparison.
- **Genesis Imagery** (`frontend/genesis.html`): the satellite-based Genesis experiment, with archived GridSat infrared sequences and its exploratory CNN–GRU metrics.

The views present separate archived artifacts; no model results are shared across components.

## Repository layout

```text
src/                         Data preparation, models, evaluation, and exporters
tests/                       Unit and integration-style tests
frontend/                    Spatial Trajectory, Monthly Activity, and Genesis Imagery views
results/                     Published experiment outputs and audits
results/track_only_*/        Storm-track forecasting artifacts
results/monthly_activity_ts/ Monthly cyclone-activity time-series artifacts
requirements.txt             Python dependencies
Makefile                     Reproduction, export, and test commands
```

Key modules include `src/ibtracs.py` (filtering and chronological samples), `src/baselines.py` and `src/gru.py` (storm-track models), `src/monthly_activity*.py` (monthly models), and `src/genesis_gridsat_gru.py` (the Genesis adapter and CNN–GRU).

## Run and reproduce

Create an environment, install dependencies, and set the external IBTrACS location:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export IBTRACS_CSV=/path/to/ibtracs.NA.list.v04r01.csv
```

Run the test suite:

```bash
make test
```

Reproduce the primary storm-track outputs:

```bash
IBTRACS_CSV="$IBTRACS_CSV" make reproduce-track-only
```

Run the monthly cyclone-activity sequence:

```bash
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity-xgboost
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity-lstm
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity-prophet
make monthly-activity-ensemble
make monthly-frontend-data
```

Run the Genesis sanity pass with prepared external acquisition state and crops:

```bash
python -m src.run_genesis_gridsat_gru_sanity \
  --manifest results/genesis_candidate_manifest.csv \
  --state-dir /path/to/genesis_gridsat_state \
  --crop-root /path/to/genesis_gridsat_crops \
  --limit 2
```

The production Genesis runner accepts the same three paths plus `--cache-dir` and `--output-dir`; see `python -m src.run_genesis_gridsat_gru --help`. To browse the committed frontend artifacts locally:

```bash
python3 -m http.server 8000 --directory frontend
```

Open `http://localhost:8000/`.

## Scope and limitations

- The storm-track and monthly evaluations are retrospective North Atlantic analyses over fixed historical splits; the Genesis experiment is a separate multi-basin cohort.
- Storm-track models use track and maximum-wind history only; they do not evaluate environmental predictors or real-time operational workflows.
- The monthly target is a retained-storm activity count, not a physical genesis label; its 72-month test window is one historical evaluation period.
- The Genesis CNN–GRU uses a separate frozen cohort and satellite archive. Its modest ROC-AUC and comparison with the majority baseline motivate treating it as exploratory.
