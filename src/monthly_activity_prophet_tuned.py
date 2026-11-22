"""Fine-tuned Prophet for monthly storm-activity forecasts.

Widens the frozen 4-point Prophet grid to a seeded random search over
changepoint prior/range, seasonality prior, Fourier order and seasonality mode
(additive / multiplicative).  Selection uses validation months only.  Prophet
is fit on the zero-inflated count with no external regressors, like the frozen
Prophet experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from .monthly_activity_common import (
    DEFAULT_RESULTS_DIR,
    forecast_months,
    load_monthly_series,
    png_forecast_plot,
    split_frame,
    validation_months,
    write_model_outputs,
)

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def candidates(n_trials: int, seed: int = 42) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    space = {
        "changepoint_prior_scale": (0.001, 0.01, 0.05, 0.1, 0.3),
        "changepoint_range": (0.8, 0.95),
        "seasonality_prior_scale": (1.0, 5.0, 10.0, 20.0),
        "yearly_fourier_order": (3, 5, 8),
        "seasonality_mode": ("additive", "multiplicative"),
    }
    out = [{"changepoint_prior_scale": 0.1, "changepoint_range": 0.8, "seasonality_prior_scale": 10.0, "yearly_fourier_order": 5, "seasonality_mode": "additive"}]
    seen = {tuple(sorted(out[0].items()))}
    while len(out) < n_trials:
        pick = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = tuple(sorted(pick.items()))
        if key not in seen:
            seen.add(key)
            out.append(pick)
    return out


def fit_predict(series: pd.Series, target: pd.Timestamp, params: dict[str, object]) -> float:
    history = series.loc[:target - pd.offsets.MonthBegin(1)]
    frame = pd.DataFrame({"ds": history.index.tz_localize(None), "y": history.to_numpy(dtype=float)})
    model = Prophet(growth="linear", yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode=str(params["seasonality_mode"]), changepoint_prior_scale=float(params["changepoint_prior_scale"]),
                    changepoint_range=float(params["changepoint_range"]), seasonality_prior_scale=float(params["seasonality_prior_scale"]), uncertainty_samples=0)
    model.add_seasonality(name="yearly", period=365.25, fourier_order=int(params["yearly_fourier_order"]))
    model.fit(frame)
    return float(model.predict(pd.DataFrame({"ds": [target.tz_localize(None)]}))["yhat"].iloc[0])


def expanding(series: pd.Series, months: pd.DatetimeIndex, params: dict[str, object]) -> pd.Series:
    return pd.Series([fit_predict(series, m, params) for m in months], index=months, name="forecast")


def run_experiment(output_dir: str | Path, n_trials: int = 10) -> dict[str, object]:
    output = Path(output_dir)
    series = load_monthly_series(output)
    val = validation_months(series)
    actual = series.loc[val].to_numpy()
    trials = []
    for i, params in enumerate(candidates(n_trials)):
        pred = expanding(series, val, params).to_numpy()
        trials.append({**params, "validation_mae": float(np.mean(np.abs(pred - actual))), "validation_rmse": float(np.sqrt(np.mean((pred - actual) ** 2)))})
        pd.DataFrame(trials).to_csv(output / "prophet_tuned_search.csv", index=False)
        print(f"  trial {i + 1}/{n_trials} val MAE={trials[-1]['validation_mae']:.3f}", flush=True)
    best = min(trials, key=lambda r: (r["validation_mae"], r["validation_rmse"]))
    params = {k: best[k] for k in trials[0] if not k.startswith("validation_")}
    frame = split_frame(series, expanding(series, forecast_months(series), params), "prophet_tuned")
    rows = write_model_outputs(output, series, frame, "prophet_tuned", {"selected_hyperparameters": params, "search_trials": len(trials), "selection_metric": "validation MAE"})
    png_forecast_plot(output / "prophet_tuned_forecast_vs_actual.png", frame, "Tuned Prophet one-step forecast vs actual", "#76b7b2")
    return {"model": "prophet_tuned", "selected_hyperparameters": params, "metrics": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output_dir, args.trials), indent=2, default=str))


if __name__ == "__main__":
    main()
