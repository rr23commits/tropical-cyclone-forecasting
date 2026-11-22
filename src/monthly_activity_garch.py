"""ARCH / GARCH monthly storm-activity forecasts with conditional-variance intervals.

Mean equation:  y_t = c + phi * y_{t-1} (optional) + sum_k b_k * harmonic_k(month_t) + e_t
Variance:       ARCH(1) or GARCH(1,1), normal or Student-t innovations.

Storm counts are strongly heteroskedastic (variance rises with the seasonal
mean and clusters in active years), so the conditional variance matters even
though the point forecast is the mean equation.  Besides point accuracy this
module therefore scores 95% prediction intervals: GARCH intervals vs. a
constant-variance (homoskedastic) interval from the same mean model.

Protocol: expanding-window one-step-ahead, configuration chosen on validation
months only.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

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

Z95 = 1.959964


def harmonics(months: pd.DatetimeIndex, order: int) -> np.ndarray:
    angle = 2 * np.pi * months.month.to_numpy() / 12.0
    return np.column_stack([f(k * angle) for k in range(1, order + 1) for f in (np.sin, np.cos)])


def candidate_grid() -> list[dict[str, object]]:
    return [{"ar_lags": lag, "harmonics": order, "p": p, "q": q, "dist": dist}
            for lag in (0, 1) for order in (2, 3) for (p, q) in ((1, 0), (1, 1)) for dist in ("normal", "t")]


def _design(series: pd.Series, end_pos: int, order: int) -> np.ndarray:
    return harmonics(series.index[:end_pos], order)


def fit_one_step(series: pd.Series, target_month: pd.Timestamp, cfg: dict[str, object]) -> tuple[float, float, float]:
    """Return (mean forecast, conditional sigma forecast, constant-variance sigma)."""
    pos = series.index.get_loc(target_month)
    y = series.to_numpy(dtype=float)[:pos]
    x = _design(series, pos, int(cfg["harmonics"]))
    x_target = harmonics(pd.DatetimeIndex([target_month]), int(cfg["harmonics"]))[0]
    lags = int(cfg["ar_lags"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(y, x=x, mean="ARX", lags=lags if lags else None, vol="GARCH", p=int(cfg["p"]), o=0, q=int(cfg["q"]), dist=str(cfg["dist"]), rescale=False)
        result = model.fit(disp="off", show_warning=False)
    params = result.params
    names = list(params.index)
    mean = float(params["Const"])
    if lags:
        mean += float(params["y[1]"]) * y[-1]
    mean += sum(float(params[n]) * x_target[int(n[1:])] for n in names if n.startswith("x") and n[1:].isdigit())
    forecast_x = {f"x{j}": np.array([[x_target[j]]]) for j in range(len(x_target))}
    variance = float(result.forecast(horizon=1, x=forecast_x, start=len(y) - 1, reindex=False).variance.to_numpy()[-1, 0])
    constant_sigma = float(np.std(np.asarray(result.resid)[~np.isnan(np.asarray(result.resid))]))
    return mean, float(np.sqrt(max(variance, 1e-9))), constant_sigma


def expanding(series: pd.Series, months: pd.DatetimeIndex, cfg: dict[str, object]) -> pd.DataFrame:
    rows = [fit_one_step(series, m, cfg) for m in months]
    return pd.DataFrame(rows, index=months, columns=["forecast", "sigma", "sigma_const"])


def interval_scores(actual: np.ndarray, forecast: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    lo, hi = forecast - Z95 * sigma, forecast + Z95 * sigma
    inside = (actual >= lo) & (actual <= hi)
    alpha = 0.05
    winkler = (hi - lo) + (2 / alpha) * np.maximum(lo - actual, 0) + (2 / alpha) * np.maximum(actual - hi, 0)
    return {"coverage_95": float(inside.mean() * 100), "mean_width": float((hi - lo).mean()), "winkler": float(winkler.mean())}


def run_experiment(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    series = load_monthly_series(output)
    val = validation_months(series)
    trials = []
    for i, cfg in enumerate(candidate_grid()):
        pred = expanding(series, val, cfg)
        actual = series.loc[val].to_numpy()
        trials.append({**cfg, "validation_mae": float(np.mean(np.abs(pred.forecast - actual))), "validation_rmse": float(np.sqrt(np.mean((pred.forecast - actual) ** 2)))})
        print(f"  config {i + 1}/{len(candidate_grid())} val MAE={trials[-1]['validation_mae']:.3f}", flush=True)
    pd.DataFrame(trials).to_csv(output / "garch_search.csv", index=False)
    best = min(trials, key=lambda r: (r["validation_mae"], r["validation_rmse"]))
    cfg = {k: best[k] for k in candidate_grid()[0]}
    result = expanding(series, forecast_months(series), cfg)
    frame = split_frame(series, result.forecast, "garch")
    frame["sigma"] = result.sigma.to_numpy()
    frame["sigma_const"] = result.sigma_const.to_numpy()
    frame["lower95"] = frame.forecast - Z95 * frame.sigma
    frame["upper95"] = frame.forecast + Z95 * frame.sigma

    # Volatility diagnostics on the training window only.
    train = series.loc[:TRAIN_END]
    x_train = harmonics(train.index, int(cfg["harmonics"]))
    design = np.column_stack([np.ones(len(train)), x_train])
    coef, *_ = np.linalg.lstsq(design, train.to_numpy(), rcond=None)
    resid = train.to_numpy() - design @ coef
    lm_stat, lm_p, _, _ = het_arch(resid, nlags=6)
    month_sigma = frame.groupby(pd.to_datetime(frame.month).dt.month).sigma.mean()
    intervals = {}
    for split in ("validation", "test"):
        part = frame[frame.split == split]
        intervals[split] = {"garch": interval_scores(part.actual.to_numpy(), part.forecast.to_numpy(), part.sigma.to_numpy()),
                            "constant_variance": interval_scores(part.actual.to_numpy(), part.forecast.to_numpy(), part.sigma_const.to_numpy())}
    audit = {"selected_configuration": cfg, "search_trials": len(trials), "selection_metric": "validation MAE",
             "arch_lm_test_train_residuals": {"lags": 6, "statistic": float(lm_stat), "p_value": float(lm_p)},
             "prediction_intervals_95": intervals,
             "mean_conditional_sigma_by_month": {int(k): float(v) for k, v in month_sigma.items()}}
    rows = write_model_outputs(output, series, frame, "garch", audit)
    png_forecast_plot(output / "garch_forecast_vs_actual.png", frame, "GARCH one-step forecast with 95% conditional interval", "#59a14f", (frame.lower95.to_numpy(), frame.upper95.to_numpy()))
    return {"model": "garch", "selected_configuration": cfg, "metrics": rows, "intervals": intervals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
