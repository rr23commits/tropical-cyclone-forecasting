# Phase 2 — Baselines and Evaluation

## Goal

Establish leakage-safe, held-out baseline results for +6, +24, and +48-hour track and wind forecasts.

## Implementation

`src/baselines.py` provides persistence, constant-motion, and training-only standardized ridge regression. `src/evaluation.py` reconstructs predicted geographic states, calculates Haversine track error and wind MAE/RMSE, and applies whole-storm split labels. `python3 -m src.run_baselines <csv>` runs all horizons and writes `results/phase2_baselines.csv`.

## Verification

Tests cover baseline predictions, direct ridge fitting, perfect reconstruction metrics, and storm-atomic sample splitting. The real audited CSV was evaluated with training through 2015, validation through 2019, and testing from 2020 onward.

## Limitations

The pipeline is deterministic and has no neural, ERA5, pressure, or satellite component.
