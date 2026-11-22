"""Recurrent (GRU / LSTM) monthly storm-activity forecasts: baseline and tuned.

``--mode baseline`` reproduces the frozen small-LSTM design with a GRU cell
(univariate 12-month history, MSE loss, 4-point grid), so GRU and LSTM are an
apples-to-apples pair.

``--mode tuned`` is the fine-tuning pass: a seeded random search over lookback,
width, depth, dropout, weight decay, learning rate, epochs, calendar covariates
and loss (MSE vs. Poisson).  Hyperparameters are selected on validation months
only; the selected model is then refit every month on an expanding window and
averaged over several seeds.  Test months never influence selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .monthly_activity_common import (
    DEFAULT_RESULTS_DIR,
    forecast_months,
    load_monthly_series,
    png_forecast_plot,
    split_frame,
    validation_months,
    write_model_outputs,
)

torch.set_num_threads(1)
SEED = 42
BASELINE_LOOKBACK = 12


class Recurrent(nn.Module):
    def __init__(self, cell: str, input_size: int, hidden: int, layers: int, dropout: float, extra: int):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(input_size, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + extra, 1)

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        out, _ = self.rnn(x)
        h = self.drop(out[:, -1, :])
        if extra is not None:
            h = torch.cat([h, extra], dim=1)
        return self.head(h)


def _calendar(months: pd.DatetimeIndex) -> np.ndarray:
    angle = 2 * np.pi * months.month.to_numpy() / 12.0
    return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def build_samples(series: pd.Series, target_pos: int, lookback: int, covariates: bool):
    """Supervised windows whose targets all lie strictly before ``target_pos``.

    Returns (x_train, extra_train, y_train, x_target, extra_target).  The
    target-month window uses only observations before the target month.
    """
    values = series.to_numpy(dtype=np.float32)
    cal = _calendar(series.index)
    history = values[:target_pos]
    windows = np.lib.stride_tricks.sliding_window_view(history, lookback)  # window k -> target k+lookback
    x_all = windows.copy()
    train_x, train_y = x_all[:-1], history[lookback:]
    target_x = x_all[-1:]
    train_pos = np.arange(lookback, target_pos)
    cal_seq = np.lib.stride_tricks.sliding_window_view(cal[:target_pos], (lookback, 2))[:, 0]  # (n, lookback, 2)
    extra_train = cal[train_pos] if covariates else None
    extra_target = cal[target_pos:target_pos + 1] if covariates else None
    return train_x, cal_seq[:-1], cal_seq[-1:], train_y, target_x, extra_train, extra_target


def _fit_predict(series: pd.Series, target_month: pd.Timestamp, params: dict[str, object]) -> float:
    target_pos = series.index.get_loc(target_month)
    lookback = int(params["lookback"])
    covariates = bool(params["covariates"])
    train_x, cal_train, cal_target, train_y, target_x, extra_train, extra_target = build_samples(series, target_pos, lookback, covariates)
    history = series.to_numpy(dtype=np.float32)[:target_pos]
    mean, scale = float(history.mean()), float(history.std()) or 1.0
    def prep(x, cal):
        x = ((x - mean) / scale)[..., None]
        return torch.from_numpy(np.concatenate([x, cal], axis=2).astype(np.float32) if covariates else x.astype(np.float32))
    x_train, x_target = prep(train_x, cal_train), prep(target_x, cal_target)
    e_train = torch.from_numpy(extra_train) if covariates else None
    e_target = torch.from_numpy(extra_target) if covariates else None
    poisson = params["loss"] == "poisson"
    y_train = torch.from_numpy(train_y.reshape(-1, 1) if poisson else ((train_y - mean) / scale).reshape(-1, 1))
    predictions = []
    for seed in range(int(params["n_seeds"])):
        torch.manual_seed(int(params["seed"]) + seed)
        model = Recurrent(str(params["cell"]), x_train.shape[2], int(params["hidden_size"]), int(params["layers"]), float(params["dropout"]), 2 if covariates else 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
        loss_fn = nn.PoissonNLLLoss(log_input=True, full=False) if poisson else nn.MSELoss()
        model.train()
        for _ in range(int(params["epochs"])):
            optimizer.zero_grad()
            loss_fn(model(x_train, e_train), y_train).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = float(model(x_target, e_target).item())
        predictions.append(float(np.exp(np.clip(out, -10, 5))) if poisson else out * scale + mean)
    return float(np.mean(predictions))


def expanding_forecast(series: pd.Series, months: pd.DatetimeIndex, params: dict[str, object]) -> pd.Series:
    return pd.Series([_fit_predict(series, m, params) for m in months], index=months, name="forecast")


def baseline_grid(cell: str) -> list[dict[str, object]]:
    return [{"cell": cell, "hidden_size": h, "layers": 1, "dropout": 0.0, "learning_rate": lr, "weight_decay": 0.0, "epochs": 30,
             "lookback": BASELINE_LOOKBACK, "covariates": False, "loss": "mse", "seed": SEED, "n_seeds": 1}
            for h in (4, 8) for lr in (0.005, 0.01)]


def tuned_candidates(cell: str, n_trials: int, rng_seed: int = SEED) -> list[dict[str, object]]:
    """Seeded random-search candidates (the baseline configuration is always included)."""
    rng = np.random.default_rng(rng_seed)
    space = {
        "hidden_size": (8, 16, 32),
        "layers": (1, 2),
        "dropout": (0.0, 0.1, 0.3),
        "learning_rate": (0.003, 0.01, 0.02),
        "weight_decay": (0.0, 1e-4, 1e-3),
        "epochs": (40, 80, 120),
        "lookback": (12, 24),
        "covariates": (False, True),
        "loss": ("mse", "poisson"),
    }
    candidates = [baseline_grid(cell)[0]]
    seen = set()
    while len(candidates) < n_trials:
        pick = {key: values[int(rng.integers(len(values)))] for key, values in space.items()}
        key = tuple(sorted(pick.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"cell": cell, **pick, "seed": SEED, "n_seeds": 1})
    return candidates


def select(series: pd.Series, candidates: list[dict[str, object]], log: Path | None = None) -> tuple[dict[str, object], list[dict[str, object]]]:
    months = validation_months(series)
    actual = series.loc[months].to_numpy()
    trials = []
    for i, params in enumerate(candidates):
        prediction = expanding_forecast(series, months, params).to_numpy()
        row = {**params, "validation_mae": float(np.mean(np.abs(prediction - actual))), "validation_rmse": float(np.sqrt(np.mean((prediction - actual) ** 2)))}
        trials.append(row)
        if log:
            pd.DataFrame(trials).to_csv(log, index=False)
        print(f"  trial {i + 1}/{len(candidates)} val MAE={row['validation_mae']:.3f} RMSE={row['validation_rmse']:.3f}", flush=True)
    best = min(trials, key=lambda r: (r["validation_mae"], r["validation_rmse"]))
    params = {k: best[k] for k in candidates[0]}
    return params, trials


def run_experiment(output_dir: str | Path, cell: str, mode: str, n_trials: int = 12, final_seeds: int = 3) -> dict[str, object]:
    output = Path(output_dir)
    series = load_monthly_series(output)
    name = cell if mode == "baseline" else f"{cell}_tuned"
    candidates = baseline_grid(cell) if mode == "baseline" else tuned_candidates(cell, n_trials)
    params, trials = select(series, candidates, output / f"{name}_search.csv")
    if mode == "tuned":
        params = {**params, "n_seeds": final_seeds}
    months = forecast_months(series)
    frame = split_frame(series, expanding_forecast(series, months, params), name)
    rows = write_model_outputs(output, series, frame, name, {"selected_hyperparameters": params, "search_trials": len(trials), "selection_metric": "validation MAE", "mode": mode})
    png_forecast_plot(output / f"{name}_forecast_vs_actual.png", frame, f"{name.upper()} one-step forecast vs actual")
    return {"model": name, "selected_hyperparameters": params, "metrics": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=["gru", "lstm"], required=True)
    parser.add_argument("--mode", choices=["baseline", "tuned"], default="baseline")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--final-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output_dir, args.cell, args.mode, args.trials, args.final_seeds), indent=2, default=str))


if __name__ == "__main__":
    main()
