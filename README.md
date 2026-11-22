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

Models include seasonal naive, ARIMA, SARIMA, additive Holt–Winters, XGBoost with past-only features, a small LSTM, Prophet, AutoTS, and a validation-weighted XGBoost–Prophet ensemble.

| Held-out model | MAE | RMSE | MASE | WAPE (%) |
|---|---:|---:|---:|---:|
| Seasonal naive | 1.181 | 2.017 | 1.083 | 70.833 |
| ARIMA | 1.528 | 2.392 | 1.401 | 91.667 |
| SARIMA | 1.181 | 2.017 | 1.083 | 70.833 |
| Additive Holt–Winters | 2.093 | 2.606 | 1.920 | 125.594 |
| XGBoost | 1.010 | 1.506 | 0.926 | 60.575 |
| LSTM | 1.254 | 1.947 | 1.150 | 75.221 |
| Prophet | 1.030 | 1.431 | 0.945 | 61.803 |
| AutoTS | 1.097 | 2.017 | 1.006 | 65.833 |
| XGBoost–Prophet ensemble | 1.016 | 1.461 | 0.931 | 60.937 |

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

### Advanced monthly models, fine-tuning and comparison

Same series and protocol as above (train 1980-2015, validation 2016-2019, test 2020-2025, expanding-window one-step forecasts, hyperparameters chosen on validation only). New models run from the archived `monthly_series.csv`, so the raw IBTrACS file is not needed: `make monthly-activity-advanced`, then `make monthly-activity-compare`.

| Requirement | Implementation |
|---|---|
| LSTM / GRU | `src/monthly_activity_rnn.py` (GRU baseline; tuned GRU and LSTM with lookback, width, depth, dropout, weight decay, calendar covariates, MSE vs Poisson loss, seed averaging) |
| XGBoost | existing, plus `monthly_activity_xgboost_tuned.py` (richer past-only features, 60-trial search, Poisson objective option) |
| Prophet | existing, plus `monthly_activity_prophet_tuned.py` (8-trial search incl. multiplicative seasonality) |
| ARCH / GARCH | `monthly_activity_garch.py` (seasonal-harmonic mean + ARCH(1)/GARCH(1,1), normal/t, 95% conditional intervals) |
| VAR | `monthly_activity_var.py` (VARX on a first-half / second-half-of-month split of the count; `--exog` accepts real climate drivers) |
| Time-series Transformer | `monthly_activity_transformer.py` (PatchTST-style, univariate; 6-trial search, 2-seed average) |
| Metrics, table, plots | `monthly_activity_compare.py` -> `results/monthly_activity_ts/comparison_*` (RMSE, MAE, MAPE, sMAPE, MASE, WAPE, bias; Diebold-Mariano tests; error by season) |

Full tables: `results/monthly_activity_ts/comparison_table.md`. Figures: `comparison_error_bars.png`, `comparison_test_overlay.png`, `comparison_error_by_season.png`, `comparison_tuning_gain.png`, `comparison_validation_vs_test.png`.

Held-out test (72 months), sorted by RMSE:

| Model | RMSE | MAE | MAPE % (y>0) |
|---|---:|---:|---:|
| Prophet | 1.431 | 1.030 | 51.4 |
| Prophet (tuned) | 1.459 | 1.043 | 52.2 |
| XGBoost+Prophet ensemble | 1.461 | 1.016 | 51.5 |
| LSTM (tuned) | 1.477 | 0.984 | 56.1 |
| GRU (tuned) | 1.502 | 0.998 | 57.1 |
| VARX (half-month) | 1.503 | 1.022 | 56.8 |
| XGBoost | 1.506 | 1.010 | 51.8 |
| XGBoost (tuned) | 1.523 | 0.976 | 53.5 |
| ARX-GARCH | 1.536 | 1.012 | 56.7 |
| PatchTST | 1.560 | 1.026 | 50.5 |
| LSTM | 1.947 | 1.254 | 60.1 |
| Seasonal naive / SARIMA | 2.017 | 1.181 | 78.9 |
| GRU | 2.050 | 1.401 | 59.6 |
| ARIMA | 2.392 | 1.528 | 94.5 |
| Holt-Winters | 2.606 | 2.093 | 92.4 |

**Metric notes.** The series is zero in 304 of 552 months, so MAPE is computed only on months with at least one storm (33 of 72 test months). sMAPE is reported but is dominated by a zero-month artifact (any small non-zero forecast in a zero month scores 200%), so it penalises smooth models and should not be used to rank them. RMSE and MAE are the primary metrics.

**Fine-tuning.** Validation-selected tuning cut test RMSE by about 24% for LSTM and 27% for GRU (mainly from a 24-month lookback and calendar covariates), but changed XGBoost by only +1% (RMSE; MAE -3%) and made Prophet slightly worse on test (+2% RMSE) despite a better validation score, a sign of overfitting the 48 validation months. PatchTST validation MAE (0.80) was clearly worse than the other models' (0.60-0.65) and its test MAE ended near the pack; with ~450 training points it has little room to beat simpler models.

**Statistical caution.** Against seasonal naive, Diebold-Mariano tests (squared error) give p of about 0.03-0.05 for Prophet, tuned Prophet, the ensemble and tuned LSTM, and about 0.06-0.09 for XGBoost, VARX, tuned GRU, GARCH and PatchTST. Differences among the top ten models are not significant (Diebold-Mariano of Prophet vs. each of them gives p between 0.06 and 0.39). Treat the ranking inside the top group as a tie; the robust finding is that seasonality-aware learned models beat ARIMA, Holt-Winters and the naive baselines.

### Interpretation for the real-world domain

* **Peak season is where it matters and where models fail.** On the test set the off-season (Dec-May, 36 months) is nearly trivial (MAE 0.2-0.4 for all models). Errors concentrate in Aug-Oct (MAE about 1.8-2.1) and Jun-Jul (about 1.5-1.8). The learned models under-forecast in these months (peak-season bias -0.5 to -1.1 storms/month for the top models, larger for the untuned LSTM/GRU), whereas seasonal naive is less biased (+0.33) but noisier. Month-ahead counts are not precise enough to plan for individual months; they are better read as seasonal-level guidance.
* **Volatility is seasonal and clustered.** An ARCH-LM test on training residuals rejects constant variance (p about 2e-6). GARCH conditional sigma is about 0.6 in Jan-May, rises to about 1.5-1.8 in Jul-Sep, and peaks at about 2.3 in October. The practical implication is that uncertainty bands should widen in the active months. GARCH 95% intervals covered 88.9% of test months vs. 84.7% for a constant-variance band, still below nominal, and were wider with a slightly worse Winkler score (11.3 vs 10.6), so the calibration gain is modest.
* **Little extra signal from clustering.** The VAR first-half/second-half-of-month Granger tests were not significant (p = 0.12, 0.60), and VARX performed like the univariate models. Gains beyond the seasonal cycle and recent history probably need external drivers (ENSO, Atlantic SST, vertical wind shear), which this repository does not contain.
* **Operational reading.** Use the model forecast as a prior on monthly activity and the GARCH band for planning headroom in Jul-Oct. Do not use these counts for storm-level or landfall decisions; that is what the track-forecasting component addresses.
* **Limits.** One test window of 72 months, a single series from a reanalysed record, counts defined by first retained observation (not physical genesis), and no climate covariates.

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
IBTRACS_CSV="$IBTRACS_CSV" make monthly-activity-autots
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
