"""Shared helpers for the advanced monthly storm-activity models.

The advanced models (GRU, GARCH, VAR, PatchTST, tuned variants) all follow the
frozen monthly protocol in ``monthly_activity``: chronological splits
(train 1980-2015, validation 2016-2019, test 2020-2025) and expanding-window
one-step-ahead forecasts.  This module only adds what they share:

* a loader that reads the archived ``monthly_series.csv`` (so these models run
  without the raw IBTrACS CSV, and are guaranteed to use the identical series),
* a metric set that adds MAPE / sMAPE / bias to the frozen MAE / RMSE / MASE /
  WAPE, and
* a common writer for forecasts, metrics, audit JSON and plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .monthly_activity import END, START, TRAIN_END, VALIDATION_END, split_label

DEFAULT_RESULTS_DIR = Path("results/monthly_activity_ts")

SEASONS = {
    "off_season (Dec-May)": (12, 1, 2, 3, 4, 5),
    "early_season (Jun-Jul)": (6, 7),
    "peak_season (Aug-Oct)": (8, 9, 10),
    "late_season (Nov)": (11,),
}


def season_of(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise ValueError(month)


def load_monthly_series(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.Series:
    """Load the archived, zero-filled monthly storm-count series."""
    frame = pd.read_csv(Path(results_dir) / "monthly_series.csv")
    index = pd.DatetimeIndex(pd.to_datetime(frame["month"], utc=True))
    series = pd.Series(frame["storm_count"].to_numpy(dtype=float), index=index, name="storm_count")
    expected = pd.date_range(START, END, freq="MS", tz="UTC")
    if len(series) != len(expected) or not series.index.equals(expected):
        raise ValueError("monthly_series.csv does not match the frozen 1980-01..2025-12 monthly index")
    return series


def validation_months(series: pd.Series) -> pd.DatetimeIndex:
    return series.index[(series.index > TRAIN_END) & (series.index <= VALIDATION_END)]


def forecast_months(series: pd.Series) -> pd.DatetimeIndex:
    return series.index[series.index > TRAIN_END]


def mape_nonzero(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, int]:
    """MAPE over months with at least one storm.

    Storm counts are zero in 304 of 552 months, so classical MAPE is undefined
    for most of the series.  We therefore report MAPE *conditional on y > 0* and
    always report how many months it covers.
    """
    mask = np.asarray(actual) > 0
    if not mask.any():
        return float("nan"), 0
    return float(np.mean(np.abs(forecast[mask] - actual[mask]) / actual[mask]) * 100), int(mask.sum())


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE in percent; months where actual and forecast are both zero count as 0 error."""
    denominator = np.abs(actual) + np.abs(forecast)
    ratio = np.divide(2 * np.abs(forecast - actual), denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    return float(ratio.mean() * 100)


def extended_metrics(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray, train: pd.Series) -> dict[str, float | int]:
    """MAE, RMSE, MAPE (y>0), sMAPE, MASE, WAPE and bias for one split."""
    y = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    error = f - y
    scale = float(np.mean(np.abs(np.diff(train.to_numpy()))))
    mape, n_positive = mape_nonzero(y, f)
    total = float(y.sum())
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": mape,
        "mape_n": n_positive,
        "smape": smape(y, f),
        "mase": float(np.mean(np.abs(error)) / scale) if scale else float("nan"),
        "wape": float(np.abs(error).sum() / total * 100) if total else float("nan"),
        "bias": float(error.mean()),
    }


def diebold_mariano(actual: np.ndarray, forecast_a: np.ndarray, forecast_b: np.ndarray, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano test for equal one-step accuracy (A vs B).

    Returns (statistic, two-sided p-value).  A negative statistic means model A
    has the lower loss.  Uses the Harvey-Leybourne-Newbold small-sample
    correction with a t(n-1) reference distribution; h = 1 so no HAC lags.
    """
    from scipy import stats

    d = np.abs(np.asarray(forecast_a) - actual) ** power - np.abs(np.asarray(forecast_b) - actual) ** power
    n = len(d)
    variance = d.var(ddof=0) / n
    if variance <= 0:
        return float("nan"), float("nan")
    statistic = d.mean() / np.sqrt(variance)
    statistic *= np.sqrt((n - 1) / n)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=n - 1))


def split_frame(series: pd.Series, prediction: pd.Series, model: str) -> pd.DataFrame:
    months = prediction.index
    return pd.DataFrame({
        "month": months,
        "actual": series.loc[months].to_numpy(),
        "forecast": prediction.to_numpy(),
        "residual": series.loc[months].to_numpy() - prediction.to_numpy(),
        "model": model,
        "split": [split_label(m) for m in months],
    })


def write_model_outputs(
    output_dir: str | Path,
    series: pd.Series,
    frame: pd.DataFrame,
    model: str,
    audit: dict[str, object],
    extra_columns: bool = True,
) -> list[dict[str, object]]:
    """Write ``<model>_forecasts.csv``, ``<model>_metrics.csv`` and ``<model>_audit.json``."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{model}_forecasts.csv", index=False)
    rows = []
    for split in ("validation", "test"):
        subset = frame[frame["split"] == split]
        rows.append({"model": model, "split": split, **extended_metrics(subset.actual, subset.forecast, series.loc[:TRAIN_END])})
    pd.DataFrame(rows).to_csv(output / f"{model}_metrics.csv", index=False)
    payload = {"model": model, "forecast_months": int(len(frame)), "validation_months": int((frame.split == "validation").sum()),
               "test_months": int((frame.split == "test").sum()), **audit}
    (output / f"{model}_audit.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return rows


def png_forecast_plot(path: Path, frame: pd.DataFrame, title: str, color: str = "#4e79a7", band: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.6))
    months = pd.to_datetime(frame["month"]).dt.tz_localize(None) if getattr(pd.to_datetime(frame["month"]).dt, "tz", None) else pd.to_datetime(frame["month"])
    ax.plot(months, frame.actual, color="#111111", lw=1.4, label="actual")
    ax.plot(months, frame.forecast, color=color, lw=1.4, label="forecast")
    if band is not None:
        ax.fill_between(months, band[0], band[1], color=color, alpha=0.18, label="95% interval")
    ax.axvline(pd.Timestamp("2020-01-01"), color="#888888", ls="--", lw=1)
    ax.set_title(title)
    ax.set_ylabel("storms / month")
    ax.legend(loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
