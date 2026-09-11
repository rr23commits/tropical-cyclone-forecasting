# Project Design

## 1. Objective

Build and evaluate a short-range forecast for an already identified North Atlantic tropical cyclone. From its recent best-track history, predict its position and maximum sustained wind 6, 12, 24, and 48 hours ahead. This is a multivariate time-series task: each forecast depends on the ordered evolution of location and intensity, and the model must generalize to storms it has never seen.

## 2. Dataset

Use the official NOAA IBTrACS v04r01 North Atlantic basin CSV. Keep records with `BASIN == NA`, `TRACK_TYPE == main`, `SEASON` from 1980 through 2025, and `USA_AGENCY == hurdat_atl`. Retain exact 00:00, 06:00, 12:00, and 18:00 UTC observations with valid `USA_LAT`, `USA_LON`, and `USA_WIND`; exclude duplicate or irregular-timestamp sequences. Use eligible tropical/subtropical USA status records (`TD`, `TS`, `HU`, `TY`, `ST`, `TC`, `SD`, `SS`) at both issue and target times.

`USA_PRES` is optional because it has more missing values than the core variables. Keep `SID`, `NAME`, `ISO_TIME`, `BASIN`, `USA_STATUS`, `USA_AGENCY`, and `IFLAG` for grouping, filtering, and auditability, not as predictive inputs.

## 3. Forecasting Setup

The final history window is five six-hour observations: `t-24`, `t-18`, `t-12`, `t-6`, and `t`. Create separate direct forecasts for +6, +12, +24, and +48 hours; do not recursively feed a predicted state into a later horizon.

Core inputs are the historical `USA_LAT`, `USA_LON`, and `USA_WIND`. Primary targets are future position and future maximum sustained wind. Train targets as displacement from the issue position and wind change from issue wind, then reconstruct the future position and wind for reporting. `USA_PRES` may be added later as an input and/or secondary target only in complete-case experiments.

## 4. Feature Engineering

Investigate only features derivable from the retained history:

- latitude, longitude, and wind at each time step;
- consecutive displacements;
- translation speed and direction; and
- recent wind changes.

Represent longitude and displacement in a way that handles geographic wraparound, and calculate final track error using great-circle distance.

## 5. Models

Start with three non-neural baselines:

- persistence: retain current position and wind;
- constant motion: extrapolate recent motion, with persistent wind unless a simple trend variant is defined; and
- regularized lagged regression using the historical features.

The learned model is a compact one-layer GRU. It is evaluated alongside the baselines without changing the fixed feature set, targets, or chronological whole-storm splits. ERA5/CDS is frozen and satellite imagery is not implemented.

## 6. Evaluation

Split by whole storm and chronologically: training storms must precede validation and test storms, and no storm may occur in more than one split. Never randomly split rows or overlapping windows. Fit feature transforms only on training data, and construct each sample using observations available at its issue time.

Report great-circle track error (km) and wind MAE/RMSE separately at +6, +12, +24, and +48 hours. Compare models on the same eligible forecast cases and report sample and storm counts for every split and horizon.

## 7. Experiments

1. Compare persistence, constant motion, and Ridge regression.
2. Compare the compact GRU with those baselines.
3. Analyse held-out errors by storm, lead time, intensity, motion, and wind-change regime.

## 8. Scope Boundaries

The core project does not include:

- global or multi-basin forecasting;
- operational deployment or real-time data ingestion;
- full spatial ERA5 modelling;
- satellite-image fusion; or
- large Transformer, foundation, or weather models.

## 9. Final boundaries

- Use whole-storm chronological splits: 1980--2015 training, 2016--2019 validation, and 2020--2025 held-out testing.
- Inputs are historical latitude, longitude, wind, and history-derived motion features; targets are north/east displacement and wind change.
- `USA_PRES`, ERA5/CDS, satellite imagery, and other model families are outside the completed experiment.
