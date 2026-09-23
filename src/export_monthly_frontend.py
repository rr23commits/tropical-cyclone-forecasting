"""Export existing monthly experiment artifacts for the static research page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MODELS = {
    "seasonal_naive": "Seasonal naive",
    "arima": "ARIMA",
    "sarima": "SARIMA",
    "holt_winters": "Additive Holt-Winters",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "prophet": "Prophet",
    "ensemble_xgboost_prophet": "XGBoost + Prophet ensemble",
}


def records(path: Path) -> list[dict[str, object]]:
    def clean(value: object) -> object:
        if pd.isna(value):
            return None
        return value.item() if hasattr(value, "item") else value

    return [{key: clean(value) for key, value in row.items()} for row in pd.read_csv(path).to_dict("records")]


def build_payload(results_dir: str | Path = "results/monthly_activity_ts") -> dict[str, object]:
    root = Path(results_dir)
    forecast_files = {
        "seasonal_naive": "forecasts.csv",
        "xgboost": "xgboost_forecasts.csv",
        "lstm": "lstm_forecasts.csv",
        "prophet": "prophet_forecasts.csv",
        "ensemble_xgboost_prophet": "ensemble_forecasts.csv",
    }
    forecasts = {model: records(root / filename) for model, filename in forecast_files.items()}
    classical = pd.read_csv(root / "metrics.csv")
    metrics = [*classical.to_dict("records")]
    for model in ("xgboost", "lstm", "prophet"):
        metrics.extend(records(root / f"{model}_metrics.csv"))
    metrics.extend(records(root / "ensemble_metrics.csv"))
    return {
        "title": "Monthly storm activity",
        "source": "IBTrACS v04r01 North Atlantic / hurdat_atl",
        "models": MODELS,
        "series": records(root / "monthly_series.csv"),
        "decomposition": records(root / "decomposition.csv"),
        "forecasts": forecasts,
        "metrics": metrics,
        "protocol": {"train": "1980–2015", "validation": "2016–2019", "test": "2020–2025", "forecast": "Expanding-window one-step-ahead", "target": "Monthly count of retained SIDs"},
    }


def main() -> None:
    output = Path("frontend/data/monthly.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_payload(), separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
