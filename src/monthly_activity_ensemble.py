"""Validation-weighted XGBoost + Prophet monthly activity ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .monthly_activity import TRAIN_END, _metrics, _svg


WEIGHTS = tuple(np.linspace(0.0, 1.0, 21))


def load_forecasts(output_dir: str | Path) -> pd.DataFrame:
    output = Path(output_dir)
    xgb = pd.read_csv(output / "xgboost_forecasts.csv")
    prophet = pd.read_csv(output / "prophet_forecasts.csv")
    required = {"month", "actual", "split", "forecast"}
    if not required.issubset(xgb) or not required.issubset(prophet):
        raise ValueError("model forecast files have an incomplete schema")
    if not xgb.month.equals(prophet.month) or not xgb.split.equals(prophet.split) or not np.allclose(xgb.actual, prophet.actual):
        raise ValueError("XGBoost and Prophet forecasts do not share identical timestamps/actuals")
    return pd.DataFrame({"month": xgb.month, "actual": xgb.actual, "split": xgb.split, "xgboost": xgb.forecast, "prophet": prophet.forecast})


def blend_forecast(frame: pd.DataFrame, xgboost_weight: float) -> pd.Series:
    return xgboost_weight * frame.xgboost + (1.0 - xgboost_weight) * frame.prophet


def select_blend_weight(frame: pd.DataFrame, weights: tuple[float, ...] = WEIGHTS) -> float:
    validation = frame[frame.split == "validation"]
    if validation.empty:
        raise ValueError("validation forecasts are required for blend selection")
    scores = [(float(np.mean(np.abs(blend_forecast(validation, weight) - validation.actual))), weight) for weight in weights]
    return float(min(scores, key=lambda item: (item[0], abs(item[1] - 0.5)))[1])


def _bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height, left, top, right, bottom = 1100, 500, 90, 55, 25, 70
    lo, hi = 0.0, max(values) * 1.15 if values else 1.0
    slot = (width - left - right) / max(len(values), 1)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">', f'<text x="{left}" y="30" font-family="sans-serif" font-size="18">{title}</text>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#555"/>']
    for i, (label, value) in enumerate(zip(labels, values)):
        bar_height = (value - lo) * (height - top - bottom) / max(hi - lo, 1)
        x = left + i * slot + slot * 0.1
        y = height - bottom - bar_height
        body.extend([f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*.8:.1f}" height="{bar_height:.1f}" fill="#4e79a7"/>', f'<text x="{x+slot*.4:.1f}" y="{height-bottom+18}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>'])
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n")


def run_experiment(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    frame = load_forecasts(output)
    series = pd.read_csv(output / "monthly_series.csv", parse_dates=["month"]).set_index("month").storm_count
    weight = select_blend_weight(frame)
    frame["forecast"] = blend_forecast(frame, weight)
    frame["residual"] = frame.actual - frame.forecast
    frame["model"] = "ensemble_xgboost_prophet"
    frame.to_csv(output / "ensemble_forecasts.csv", index=False)
    metric_rows = []
    for split in ("validation", "test"):
        subset = frame[frame.split == split].set_index("month")
        metric_rows.append({"model": "ensemble_xgboost_prophet", "split": split, **_metrics(subset.actual, subset.forecast, series.loc[:TRAIN_END])})
    pd.DataFrame(metric_rows).to_csv(output / "ensemble_metrics.csv", index=False)
    report = {"model": "ensemble_xgboost_prophet", "xgboost_weight": weight, "prophet_weight": 1.0 - weight, "weight_grid": list(WEIGHTS), "forecast_months": len(frame), "validation_months": int((frame.split == "validation").sum()), "test_months": int((frame.split == "test").sum()), "component_forecasts": ["xgboost_forecasts.csv", "prophet_forecasts.csv"]}
    (output / "ensemble_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance = {"protocol": {"selection": "validation-only MAE", "forecast": "frozen blend weight on expanding one-step forecasts", "train_end": "2015-12", "validation_end": "2019-12", "test": "2020-01 through 2025-12"}, "components": ["XGBoost", "Prophet"], "selected_weight": report["xgboost_weight"], "source_artifacts": report["component_forecasts"]}
    (output / "ensemble_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    _plots(frame, metric_rows, output)
    return report


def _plots(frame: pd.DataFrame, ensemble_metrics: list[dict[str, object]], output: Path) -> None:
    _svg(output / "ensemble_forecast_vs_actual.svg", "XGBoost + Prophet ensemble vs actual", [(frame.actual.tolist(), "#111111"), (frame.forecast.tolist(), "#4e79a7")])
    metrics = []
    for filename in ("metrics.csv", "xgboost_metrics.csv", "lstm_metrics.csv", "prophet_metrics.csv"):
        metrics.append(pd.read_csv(output / filename))
    metrics.append(pd.DataFrame(ensemble_metrics))
    comparison = pd.concat(metrics, ignore_index=True)
    test = comparison[comparison.split == "test"].drop_duplicates("model").set_index("model")
    _bar_svg(output / "ensemble_model_comparison.svg", "Test MAE: all monthly models", list(test.index), test.mae.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/monthly_activity_ts"))
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
