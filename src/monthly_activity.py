"""Monthly North Atlantic storm-activity experiment.

This module is deliberately separate from the track-only sample builders.  It
counts each retained SID once, at the month of its first retained observation,
then evaluates one-step forecasts on one shared monthly index.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .ibtracs import filter_north_atlantic, load_ibtracs


START = pd.Timestamp("1980-01-01", tz="UTC")
END = pd.Timestamp("2025-12-01", tz="UTC")
TRAIN_END = pd.Timestamp("2015-12-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2019-12-01", tz="UTC")


def build_monthly_series(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return a complete zero-filled monthly count and its retained SID audit."""
    retained = filter_north_atlantic(data)
    first = retained.sort_values("ISO_TIME").groupby("SID", as_index=False).first()
    first["month"] = first["ISO_TIME"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    index = pd.date_range(START, END, freq="MS", tz="UTC")
    counts = first.groupby("month").size().reindex(index, fill_value=0).astype(float)
    counts.name = "storm_count"
    audit = first[["SID", "SEASON", "ISO_TIME", "month"]].sort_values(["month", "SID"]).reset_index(drop=True)
    return counts, audit


def split_label(timestamp: pd.Timestamp) -> str:
    if timestamp <= TRAIN_END:
        return "train"
    if timestamp <= VALIDATION_END:
        return "validation"
    return "test"


def arima_forecast(history: np.ndarray, order: tuple[int, int, int]) -> float:
    """Fit a proper ARIMA model to history and forecast one step."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(ARIMA(np.asarray(history, dtype=float), order=order).fit(method_kwargs={"maxiter": 50}).forecast(1)[0])
    except (ValueError, np.linalg.LinAlgError):
        return float(history[-1])


def sarima_forecast(history: np.ndarray, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> float:
    """Fit a proper seasonal ARIMA model to history and forecast one step."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(SARIMAX(np.asarray(history, dtype=float), order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50).forecast(1)[0])
    except (ValueError, np.linalg.LinAlgError):
        return float(history[-1])


def holt_winters_forecast(history: np.ndarray, alpha: float, beta: float, gamma: float, period: int = 12) -> float:
    """One-step additive Holt-Winters forecast with a fixed seasonal period."""
    y = np.asarray(history, dtype=float)
    if len(y) < 2 * period:
        return float(y[-1])
    season = y[:period] - y[:period].mean()
    level = float(y[:period].mean())
    trend = float((y[period:2*period].mean() - y[:period].mean()) / period)
    for value in y[period:]:
        old_level = level
        level = alpha * (value - season[len(season) % period]) + (1 - alpha) * (level + trend)
        trend = beta * (level - old_level) + (1 - beta) * trend
        season[len(season) % period] = gamma * (value - level) + (1 - gamma) * season[len(season) % period]
    return float(level + trend + season[len(y) % period])


def seasonal_naive(history: np.ndarray) -> float:
    return float(history[-12])


def seasonal_decomposition(series: pd.Series, period: int = 12) -> pd.DataFrame:
    """Centered moving-average trend plus training-independent seasonal index."""
    result = pd.DataFrame({"actual": series})
    result["trend"] = series.rolling(period, center=True, min_periods=period).mean()
    result["seasonal"] = series - result["trend"]
    result["residual"] = series - result["trend"] - result["seasonal"].groupby(series.index.month).transform("mean")
    return result


def _forecasts(series: pd.Series, model: str, setting: object) -> pd.Series:
    timestamps = series.index[series.index > TRAIN_END]
    values = []
    for timestamp in timestamps:
        history = series.loc[:timestamp - pd.offsets.MonthBegin(1)].to_numpy()
        if model == "seasonal_naive":
            prediction = seasonal_naive(history)
        elif model == "arima":
            prediction = arima_forecast(history, setting)
        elif model == "sarima":
            prediction = sarima_forecast(history, setting[0], setting[1])
        else:
            prediction = holt_winters_forecast(history, *setting)
        values.append(prediction)
    return pd.Series(values, index=timestamps, name="forecast")


def _select_settings(series: pd.Series) -> dict[str, object]:
    validation = series.index[(series.index > TRAIN_END) & (series.index <= VALIDATION_END)]
    candidates = candidate_orders()
    selected: dict[str, object] = {"seasonal_naive": "t-12"}
    for model, options in candidates.items():
        scores = []
        for setting in options:
            predictions = _forecasts(series.loc[:VALIDATION_END], model, setting)
            actual = series.loc[validation]
            scores.append((float(np.mean(np.abs(predictions - actual))), setting))
        selected[model] = min(scores, key=lambda item: item[0])[1]
    return selected


def candidate_orders() -> dict[str, list[object]]:
    """Return the fixed order grid; selection itself sees validation months only."""
    return {
        "arima": [(p, d, q) for d in (0, 1) for p, q in ((0, 0), (1, 0), (0, 1), (1, 1))],
        "sarima": [((p, d, q), (P, 1, Q, 12)) for d in (0, 1) for p, q, P, Q in ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 1), (1, 1, 0, 1))],
        "holt_winters": [(a, b, g) for a in (0.2, 0.5, 0.8) for b in (0.05, 0.2) for g in (0.1, 0.3, 0.6)],
    }


def _metrics(actual: pd.Series, prediction: pd.Series, train: pd.Series) -> dict[str, float | int]:
    error = prediction - actual
    scale = float(np.mean(np.abs(np.diff(train.to_numpy()))))
    return {"n": int(len(actual)), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "mase": float(np.mean(np.abs(error)) / scale) if scale else float("nan")}


def run_experiment(data: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    series, audit = build_monthly_series(data)
    series.to_csv(output / "monthly_series.csv", header=True, index_label="month")
    audit.to_csv(output / "retained_storm_audit.csv", index=False)
    decomposition = seasonal_decomposition(series)
    decomposition.to_csv(output / "decomposition.csv", index_label="month")
    settings = _select_settings(series)
    forecast_frames = []
    metric_rows = []
    for model, setting in settings.items():
        predictions = _forecasts(series, model, setting)
        frame = pd.DataFrame({"month": predictions.index, "actual": series.loc[predictions.index], "forecast": predictions, "model": model, "split": [split_label(x) for x in predictions.index]})
        forecast_frames.append(frame)
        for split in ("validation", "test"):
            subset = frame[frame["split"] == split].set_index("month")
            metric_rows.append({"model": model, "split": split, **_metrics(subset["actual"], subset["forecast"], series.loc[:TRAIN_END])})
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    forecasts.to_csv(output / "forecasts.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    audit_report = {"months": len(series), "zero_months": int((series == 0).sum()), "retained_sids": len(audit), "unique_audit_sids": int(audit.SID.nunique()), "start": str(series.index[0]), "end": str(series.index[-1]), "split_counts": {split: int(sum(split_label(x) == split for x in series.index)) for split in ("train", "validation", "test")}, "settings": {key: (list(value) if isinstance(value, tuple) else value) for key, value in settings.items()}}
    (output / "audit.json").write_text(json.dumps(audit_report, indent=2, default=str) + "\n")
    _plots(series, decomposition, forecasts, metric_rows, output)
    return audit_report


def _svg(path: Path, title: str, lines: list[tuple[list[float], str]], bars: list[float] | None = None) -> None:
    """Write a dependency-free SVG; plots are audit artifacts, not presentation UI."""
    width, height, left, top, right, bottom = 1100, 420, 70, 45, 25, 45
    values = [value for points, _ in lines for value in points] + (bars or [0])
    lo, hi = min(values), max(values)
    if hi == lo: hi = lo + 1
    def point(i, value, n):
        x = left + i * (width - left - right) / max(n - 1, 1)
        y = top + (hi - value) * (height - top - bottom) / (hi - lo)
        return f"{x:.1f},{y:.1f}"
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">', f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">{title}</text>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#555"/>']
    if bars is not None:
        slot = (width - left - right) / len(bars)
        for i, value in enumerate(bars):
            y = top + (hi - value) * (height - top - bottom) / (hi - lo)
            body.append(f'<rect x="{left+i*slot+slot*.1:.1f}" y="{y:.1f}" width="{slot*.8:.1f}" height="{height-bottom-y:.1f}" fill="#4e79a7"/>')
    for points, color in lines:
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(point(i,v,len(points)) for i,v in enumerate(points))}"/>')
    body.append('</svg>')
    path.write_text("\n".join(body) + "\n")


def _plots(series: pd.Series, decomposition: pd.DataFrame, forecasts: pd.DataFrame, metrics: list[dict[str, object]], output: Path) -> None:
    _svg(output / "monthly_count_series.svg", "Monthly retained North Atlantic storms", [(series.tolist(), "#1f77b4")])
    yearly = series.groupby(series.index.year).sum().tolist(); monthly = series.groupby(series.index.month).mean().tolist()
    _svg(output / "seasonality.svg", "Yearly activity", [(yearly, "#1f77b4")]); _svg(output / "month_of_year.svg", "Mean month-of-year activity", [], monthly)
    _svg(output / "decomposition.svg", "Decomposition: actual, trend, residual", [(decomposition.actual.tolist(), "#1f77b4"), (decomposition.trend.bfill().ffill().tolist(), "#e15759"), (decomposition.residual.bfill().ffill().tolist(), "#59a14f")])
    lines = [(frame.forecast.tolist(), color) for (_, frame), color in zip(forecasts.groupby("model"), ["#f28e2b", "#59a14f", "#e15759", "#af7aa1", "#76b7b2"])]
    actual = forecasts.drop_duplicates("month").actual.tolist(); _svg(output / "forecast_vs_actual.svg", "One-step forecasts vs actual", [(actual, "#111111")] + lines)
    test = pd.DataFrame(metrics).query("split == 'test'").set_index("model"); _svg(output / "model_comparison.svg", "Test MAE comparison", [], test.mae.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/monthly_activity_ts"))
    args = parser.parse_args()
    report = run_experiment(load_ibtracs(args.csv), args.output_dir)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
