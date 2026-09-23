"""Separate Prophet monthly storm-activity experiment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from .monthly_activity import (
    TRAIN_END,
    VALIDATION_END,
    _metrics,
    _svg,
    build_monthly_series,
    load_ibtracs,
    split_label,
)


logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def prophet_history(series: pd.Series, target_month: pd.Timestamp) -> pd.DataFrame:
    """Return Prophet's ds/y frame strictly before the target month."""
    history = series.loc[:target_month - pd.offsets.MonthBegin(1)]
    return pd.DataFrame({"ds": history.index.tz_localize(None), "y": history.to_numpy(dtype=float)})


def hyperparameter_grid() -> list[dict[str, object]]:
    return [{"changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "yearly_fourier_order": 5, "seasonality_mode": "additive"} for cps in (0.01, 0.1) for sps in (1.0, 10.0)]


def _fit_predict(series: pd.Series, target_month: pd.Timestamp, params: dict[str, object]) -> float:
    model = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=str(params["seasonality_mode"]),
        changepoint_prior_scale=float(params["changepoint_prior_scale"]),
        seasonality_prior_scale=float(params["seasonality_prior_scale"]),
        uncertainty_samples=0,
    )
    model.add_seasonality(name="yearly", period=365.25, fourier_order=int(params["yearly_fourier_order"]))
    model.fit(prophet_history(series, target_month))
    future = pd.DataFrame({"ds": [target_month.tz_localize(None)]})
    return float(model.predict(future)["yhat"].iloc[0])


def expanding_forecast(series: pd.Series, target_months: pd.DatetimeIndex, params: dict[str, object]) -> pd.Series:
    return pd.Series([_fit_predict(series, month, params) for month in target_months], index=target_months, name="forecast")


def select_hyperparameters(series: pd.Series) -> dict[str, object]:
    validation_months = series.index[(series.index > TRAIN_END) & (series.index <= VALIDATION_END)]
    scored = []
    for params in hyperparameter_grid():
        prediction = expanding_forecast(series, validation_months, params)
        scored.append((float(np.mean(np.abs(prediction - series.loc[validation_months]))), params))
    return min(scored, key=lambda item: item[0])[1]


def run_experiment(data: pd.DataFrame, output_dir: str | Path, source_path: str | Path | None = None) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    series, audit = build_monthly_series(data)
    target_months = series.index[series.index > TRAIN_END]
    params = select_hyperparameters(series)
    prediction = expanding_forecast(series, target_months, params)
    frame = pd.DataFrame({"month": target_months, "actual": series.loc[target_months], "forecast": prediction, "residual": series.loc[target_months] - prediction, "model": "prophet", "split": [split_label(month) for month in target_months]})
    frame.to_csv(output / "prophet_forecasts.csv", index=False)
    metric_rows = []
    for split in ("validation", "test"):
        subset = frame[frame["split"] == split].set_index("month")
        metric_rows.append({"model": "prophet", "split": split, **_metrics(subset.actual, subset.forecast, series.loc[:TRAIN_END])})
    pd.DataFrame(metric_rows).to_csv(output / "prophet_metrics.csv", index=False)
    report = {"model": "prophet", "forecast_months": len(frame), "validation_months": int((frame.split == "validation").sum()), "test_months": int((frame.split == "test").sum()), "selected_configuration": params, "yearly_seasonality": True, "external_regressors": [], "retained_sids": int(audit.SID.nunique())}
    (output / "prophet_audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    provenance = {"source": str(source_path) if source_path else "provided_dataframe", "protocol": {"train_end": "2015-12", "validation_end": "2019-12", "test": "2020-01 through 2025-12", "forecast": "expanding-window one-step-ahead", "timestamps": "shared monthly validation/test index"}, "selected_configuration": params, "yearly_seasonality": True, "external_regressors": []}
    (output / "prophet_provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    _plots(frame, metric_rows, output)
    return report


def _plots(frame: pd.DataFrame, metrics: list[dict[str, object]], output: Path) -> None:
    _svg(output / "prophet_forecast_vs_actual.svg", "Prophet monthly forecast vs actual", [(frame.actual.tolist(), "#111111"), (frame.forecast.tolist(), "#4e79a7")])
    _svg(output / "prophet_residuals.svg", "Prophet residuals", [(frame.residual.tolist(), "#e15759"), ([0.0] * len(frame), "#111111")])
    table = pd.DataFrame(metrics).set_index("split")
    _svg(output / "prophet_model_comparison.svg", "Prophet MAE: validation vs test", [], table.mae.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/monthly_activity_ts"))
    args = parser.parse_args()
    print(json.dumps(run_experiment(load_ibtracs(args.csv), args.output_dir, args.csv), indent=2, default=str))


if __name__ == "__main__":
    main()
