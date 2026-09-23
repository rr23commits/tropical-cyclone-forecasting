"""Separate XGBoost monthly storm-activity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .monthly_activity import (
    END,
    TRAIN_END,
    VALIDATION_END,
    _metrics,
    build_monthly_series,
    load_ibtracs,
    split_label,
    _svg,
)


LAGS = (1, 2, 3, 6, 12)
TRAILING_WINDOW = 3


def feature_row(series: pd.Series, target_month: pd.Timestamp, trailing_mean: bool) -> dict[str, float]:
    """Build features using values strictly before target_month."""
    history = series.loc[:target_month - pd.offsets.MonthBegin(1)]
    if len(history) < max(LAGS):
        raise ValueError("at least 12 past monthly observations are required")
    row = {f"lag_{lag}": float(history.iloc[-lag]) for lag in LAGS}
    month = target_month.month
    row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * month / 12))
    if trailing_mean:
        row[f"trailing_mean_{TRAILING_WINDOW}"] = float(history.iloc[-TRAILING_WINDOW:].mean())
    return row


def feature_frame(series: pd.Series, target_months: pd.DatetimeIndex, trailing_mean: bool) -> pd.DataFrame:
    return pd.DataFrame([feature_row(series, month, trailing_mean) for month in target_months], index=target_months)


def hyperparameter_grid() -> list[dict[str, object]]:
    return [
        {"n_estimators": estimators, "max_depth": depth, "learning_rate": rate, "trailing_mean": trailing}
        for estimators in (50, 100)
        for depth in (1, 2)
        for rate in (0.05, 0.1)
        for trailing in (False, True)
    ]


def _fit_predict(history_series: pd.Series, target_month: pd.Timestamp, params: dict[str, object]) -> float:
    trailing = bool(params["trailing_mean"])
    train_months = history_series.index[history_series.index >= history_series.index[0] + pd.offsets.MonthBegin(max(LAGS))]
    train_months = train_months[train_months < target_month]
    x_train = feature_frame(history_series, train_months, trailing)
    y_train = history_series.loc[train_months]
    x_target = feature_frame(history_series, pd.DatetimeIndex([target_month]), trailing)
    model = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "seed": 42,
            "nthread": 1,
            "tree_method": "hist",
        },
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=int(params["n_estimators"]),
        verbose_eval=False,
    )
    return float(model.predict(xgb.DMatrix(x_target))[0])


def expanding_forecast(series: pd.Series, target_months: pd.DatetimeIndex, params: dict[str, object]) -> pd.Series:
    """Refit on all available past targets before every one-step forecast."""
    predictions = [_fit_predict(series, month, params) for month in target_months]
    return pd.Series(predictions, index=target_months, name="forecast")


def select_hyperparameters(series: pd.Series) -> dict[str, object]:
    validation_months = series.index[(series.index > TRAIN_END) & (series.index <= VALIDATION_END)]
    scores = []
    for params in hyperparameter_grid():
        prediction = expanding_forecast(series, validation_months, params)
        actual = series.loc[validation_months]
        scores.append((float(np.mean(np.abs(prediction - actual))), params))
    return min(scores, key=lambda item: item[0])[1]


def run_experiment(data: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    series, audit = build_monthly_series(data)
    target_months = series.index[series.index > TRAIN_END]
    params = select_hyperparameters(series)
    prediction = expanding_forecast(series, target_months, params)
    frame = pd.DataFrame({"month": target_months, "actual": series.loc[target_months], "forecast": prediction, "residual": series.loc[target_months] - prediction, "model": "xgboost", "split": [split_label(month) for month in target_months]})
    frame.to_csv(output / "xgboost_forecasts.csv", index=False)
    metric_rows = []
    for split in ("validation", "test"):
        subset = frame[frame["split"] == split].set_index("month")
        metric_rows.append({"model": "xgboost", "split": split, **_metrics(subset.actual, subset.forecast, series.loc[:TRAIN_END])})
    pd.DataFrame(metric_rows).to_csv(output / "xgboost_metrics.csv", index=False)
    report = {
        "model": "xgboost",
        "forecast_months": len(frame),
        "validation_months": int((frame.split == "validation").sum()),
        "test_months": int((frame.split == "test").sum()),
        "feature_names": list(feature_frame(series, pd.DatetimeIndex([target_months[0]]), bool(params["trailing_mean"])).columns),
        "lags": list(LAGS),
        "trailing_mean_window": TRAILING_WINDOW,
        "selected_hyperparameters": params,
        "retained_sids": int(audit.SID.nunique()),
    }
    (output / "xgboost_audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    _plots(frame, metric_rows, output)
    return report


def _plots(frame: pd.DataFrame, metrics: list[dict[str, object]], output: Path) -> None:
    _svg(output / "xgboost_forecast_vs_actual.svg", "XGBoost monthly forecast vs actual", [(frame.actual.tolist(), "#111111"), (frame.forecast.tolist(), "#4e79a7")])
    _svg(output / "xgboost_residuals.svg", "XGBoost residuals", [(frame.residual.tolist(), "#e15759"), ([0.0] * len(frame), "#111111")])
    table = pd.DataFrame(metrics).set_index("split")
    _svg(output / "xgboost_model_comparison.svg", "XGBoost MAE: validation vs test", [], table.mae.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/monthly_activity_ts"))
    args = parser.parse_args()
    print(json.dumps(run_experiment(load_ibtracs(args.csv), args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
