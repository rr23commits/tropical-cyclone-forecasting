"""Fine-tuned XGBoost for monthly storm-activity forecasts.

Extends the frozen XGBoost experiment (5 lags + month sin/cos, 16-point grid) with:

* richer past-only features (lags up to 24, trailing means/sums over 3, 6, 12
  months, same-month-last-year, second seasonal harmonic),
* a wider seeded random search over depth, learning rate, rounds, row/column
  subsampling, min_child_weight, L2, and the objective (squared error vs. count
  Poisson),
* validation-only selection; test months are never used for selection.

All features for month t use observations strictly before t.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .monthly_activity import TRAIN_END
from .monthly_activity_common import (
    DEFAULT_RESULTS_DIR,
    forecast_months,
    load_monthly_series,
    png_forecast_plot,
    split_frame,
    validation_months,
    write_model_outputs,
)

BASE_LAGS = (1, 2, 3, 6, 12)
EXTENDED_LAGS = (1, 2, 3, 4, 5, 6, 9, 12, 18, 24)
MIN_HISTORY = 24


def feature_matrix(series: pd.Series, feature_set: str) -> pd.DataFrame:
    """Past-only features for every month (row t uses y[:t])."""
    shifted = series.shift(1)  # everything below is computed from y_{t-1} and earlier
    lags = BASE_LAGS if feature_set == "base" else EXTENDED_LAGS
    cols = {f"lag_{k}": series.shift(k) for k in lags}
    angle = 2 * np.pi * series.index.month.to_numpy() / 12.0
    cols["month_sin"], cols["month_cos"] = np.sin(angle), np.cos(angle)
    if feature_set == "extended":
        cols["month_sin2"], cols["month_cos2"] = np.sin(2 * angle), np.cos(2 * angle)
        for w in (3, 6, 12):
            cols[f"mean_{w}"] = shifted.rolling(w).mean()
        cols["sum_12"] = shifted.rolling(12).sum()
        values = series.to_numpy()
        cols["same_month_prev_years_mean"] = pd.Series(
            [float(np.mean(values[list(range(i - 12, -1, -12))[:5]])) if i >= 12 else np.nan for i in range(len(series))], index=series.index)
    return pd.DataFrame(cols, index=series.index)


def search_space_sample(rng: np.random.Generator) -> dict[str, object]:
    space = {
        "feature_set": ("base", "extended"),
        "objective": ("reg:squarederror", "count:poisson"),
        "max_depth": (1, 2, 3, 4),
        "learning_rate": (0.02, 0.05, 0.1),
        "n_estimators": (50, 100, 200, 300),
        "subsample": (0.7, 1.0),
        "colsample_bytree": (0.7, 1.0),
        "min_child_weight": (1, 3, 5),
        "reg_lambda": (1.0, 5.0, 10.0),
    }
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def candidates(n_trials: int, seed: int = 42) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    out = [{"feature_set": "base", "objective": "reg:squarederror", "max_depth": 1, "learning_rate": 0.1, "n_estimators": 100, "subsample": 1.0,
            "colsample_bytree": 1.0, "min_child_weight": 1, "reg_lambda": 1.0}]  # frozen-grid style anchor
    seen = set()
    while len(out) < n_trials:
        pick = search_space_sample(rng)
        key = tuple(sorted(pick.items()))
        if key not in seen:
            seen.add(key)
            out.append(pick)
    return out


def predict_month(features: pd.DataFrame, series: pd.Series, target: pd.Timestamp, params: dict[str, object], n_seeds: int) -> float:
    pos = series.index.get_loc(target)
    rows = np.arange(MIN_HISTORY, pos)
    x_train, y_train = features.iloc[rows], series.iloc[rows]
    x_target = features.iloc[[pos]]
    preds = []
    for seed in range(n_seeds):
        model = xgb.train(
            {"objective": params["objective"], "max_depth": int(params["max_depth"]), "learning_rate": float(params["learning_rate"]),
             "subsample": float(params["subsample"]), "colsample_bytree": float(params["colsample_bytree"]), "min_child_weight": float(params["min_child_weight"]),
             "reg_lambda": float(params["reg_lambda"]), "seed": 42 + seed, "nthread": 1, "tree_method": "hist"},
            xgb.DMatrix(x_train, label=y_train), num_boost_round=int(params["n_estimators"]), verbose_eval=False)
        preds.append(float(model.predict(xgb.DMatrix(x_target))[0]))
    return float(np.mean(preds))


def expanding(series: pd.Series, months: pd.DatetimeIndex, params: dict[str, object], n_seeds: int = 1) -> pd.Series:
    features = feature_matrix(series, str(params["feature_set"]))
    return pd.Series([predict_month(features, series, m, params, n_seeds) for m in months], index=months, name="forecast")


def run_experiment(output_dir: str | Path, n_trials: int = 60, final_seeds: int = 5) -> dict[str, object]:
    output = Path(output_dir)
    series = load_monthly_series(output)
    val = validation_months(series)
    actual = series.loc[val].to_numpy()
    trials = []
    for i, params in enumerate(candidates(n_trials)):
        pred = expanding(series, val, params).to_numpy()
        trials.append({**params, "validation_mae": float(np.mean(np.abs(pred - actual))), "validation_rmse": float(np.sqrt(np.mean((pred - actual) ** 2)))})
        if (i + 1) % 10 == 0:
            print(f"  trial {i + 1}/{n_trials} best val MAE so far={min(t['validation_mae'] for t in trials):.3f}", flush=True)
    pd.DataFrame(trials).to_csv(output / "xgboost_tuned_search.csv", index=False)
    best = min(trials, key=lambda r: (r["validation_mae"], r["validation_rmse"]))
    params = {k: best[k] for k in trials[0] if not k.startswith("validation_")}
    frame = split_frame(series, expanding(series, forecast_months(series), params, final_seeds), "xgboost_tuned")
    train_mask = (series.index >= series.index[MIN_HISTORY]) & (series.index <= TRAIN_END)
    full = feature_matrix(series, str(params["feature_set"]))
    model = xgb.train({"objective": params["objective"], "max_depth": int(params["max_depth"]), "learning_rate": float(params["learning_rate"]), "nthread": 1, "seed": 42},
                      xgb.DMatrix(full[train_mask], label=series[train_mask]), num_boost_round=int(params["n_estimators"]))
    importance = {k: float(v) for k, v in sorted(model.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])}
    rows = write_model_outputs(output, series, frame, "xgboost_tuned", {"selected_hyperparameters": params, "search_trials": len(trials), "selection_metric": "validation MAE",
                                                                         "final_seed_average": final_seeds, "feature_gain_importance_train_only": importance})
    png_forecast_plot(output / "xgboost_tuned_forecast_vs_actual.png", frame, "Tuned XGBoost one-step forecast vs actual", "#f28e2b")
    return {"model": "xgboost_tuned", "selected_hyperparameters": params, "metrics": rows, "importance": importance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--final-seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output_dir, args.trials, args.final_seeds), indent=2, default=str))


if __name__ == "__main__":
    main()
