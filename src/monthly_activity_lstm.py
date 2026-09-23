"""Separate small LSTM monthly storm-activity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .monthly_activity import (
    END,
    TRAIN_END,
    VALIDATION_END,
    _metrics,
    _svg,
    build_monthly_series,
    load_ibtracs,
    split_label,
)


SEQUENCE_LENGTH = 12
SEED = 42


def sequence_for_target(series: pd.Series, target_month: pd.Timestamp) -> np.ndarray:
    """Return exactly the 12 observations ending immediately before target_month."""
    position = series.index.get_loc(target_month)
    if position < SEQUENCE_LENGTH:
        raise ValueError("target month does not have a full 12-month history")
    return series.iloc[position - SEQUENCE_LENGTH:position].to_numpy(dtype=np.float32)


def training_sequences(series: pd.Series, target_month: pd.Timestamp, mean: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised sequences only for targets before the forecast month."""
    months = series.index[SEQUENCE_LENGTH:]
    months = months[months < target_month]
    x = np.asarray([(sequence_for_target(series, month) - mean) / scale for month in months], dtype=np.float32)
    y = ((series.loc[months].to_numpy(dtype=np.float32) - mean) / scale).reshape(-1, 1)
    return x, y


class SmallLSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(values)
        return self.output(sequence[:, -1, :])


def hyperparameter_grid() -> list[dict[str, object]]:
    return [{"hidden_size": hidden, "learning_rate": rate, "epochs": 30, "seed": SEED} for hidden in (4, 8) for rate in (0.005, 0.01)]


def _fit_predict(series: pd.Series, target_month: pd.Timestamp, params: dict[str, object]) -> float:
    history = series.loc[:target_month - pd.offsets.MonthBegin(1)].to_numpy(dtype=np.float32)
    mean = float(history.mean())
    scale = float(history.std()) or 1.0
    x_train, y_train = training_sequences(series, target_month, mean, scale)
    x_target = ((sequence_for_target(series, target_month) - mean) / scale).reshape(1, SEQUENCE_LENGTH, 1)
    torch.manual_seed(int(params["seed"]))
    model = SmallLSTM(int(params["hidden_size"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    loss_fn = nn.MSELoss()
    x_tensor = torch.from_numpy(x_train).unsqueeze(-1)
    y_tensor = torch.from_numpy(y_train)
    model.train()
    for _ in range(int(params["epochs"])):
        optimizer.zero_grad()
        loss_fn(model(x_tensor), y_tensor).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        normalized = float(model(torch.from_numpy(x_target)).item())
    return normalized * scale + mean


def expanding_forecast(series: pd.Series, target_months: pd.DatetimeIndex, params: dict[str, object]) -> pd.Series:
    predictions = [_fit_predict(series, month, params) for month in target_months]
    return pd.Series(predictions, index=target_months, name="forecast")


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
    frame = pd.DataFrame({"month": target_months, "actual": series.loc[target_months], "forecast": prediction, "residual": series.loc[target_months] - prediction, "model": "lstm", "split": [split_label(month) for month in target_months]})
    frame.to_csv(output / "lstm_forecasts.csv", index=False)
    metric_rows = []
    for split in ("validation", "test"):
        subset = frame[frame["split"] == split].set_index("month")
        metric_rows.append({"model": "lstm", "split": split, **_metrics(subset.actual, subset.forecast, series.loc[:TRAIN_END])})
    pd.DataFrame(metric_rows).to_csv(output / "lstm_metrics.csv", index=False)
    report = {"model": "lstm", "sequence_length": SEQUENCE_LENGTH, "forecast_months": len(frame), "validation_months": int((frame.split == "validation").sum()), "test_months": int((frame.split == "test").sum()), "selected_architecture": {"input_size": 1, "hidden_size": int(params["hidden_size"]), "layers": 1, "output_size": 1, "cell": "LSTM"}, "selected_hyperparameters": params, "retained_sids": int(audit.SID.nunique())}
    (output / "lstm_audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    provenance = {"source": str(source_path) if source_path else "provided_dataframe", "protocol": {"train_end": "2015-12", "validation_end": "2019-12", "test": "2020-01 through 2025-12", "forecast": "expanding-window one-step-ahead", "sequence_length": SEQUENCE_LENGTH}, "selected_architecture": report["selected_architecture"], "selected_hyperparameters": report["selected_hyperparameters"]}
    (output / "lstm_provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    _plots(frame, metric_rows, output)
    return report


def _plots(frame: pd.DataFrame, metrics: list[dict[str, object]], output: Path) -> None:
    _svg(output / "lstm_forecast_vs_actual.svg", "LSTM monthly forecast vs actual", [(frame.actual.tolist(), "#111111"), (frame.forecast.tolist(), "#4e79a7")])
    _svg(output / "lstm_residuals.svg", "LSTM residuals", [(frame.residual.tolist(), "#e15759"), ([0.0] * len(frame), "#111111")])
    table = pd.DataFrame(metrics).set_index("split")
    _svg(output / "lstm_model_comparison.svg", "LSTM MAE: validation vs test", [], table.mae.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/monthly_activity_ts"))
    args = parser.parse_args()
    print(json.dumps(run_experiment(load_ibtracs(args.csv), args.output_dir, args.csv), indent=2, default=str))


if __name__ == "__main__":
    main()
