"""PatchTST-style Transformer for monthly storm-activity forecasts.

Architecture (a compact, univariate PatchTST variant):

    past L monthly counts -> overlapping patches -> linear patch embedding
    -> learned positional embedding -> N Transformer encoder layers
    -> flatten -> [+ target-month sin/cos] -> linear head -> next-month count

Protocol is the frozen monthly protocol: expanding-window one-step-ahead, a
fresh model per forecast month using only earlier targets, hyper-parameters
chosen by a seeded random search on validation months only, final model
averaged over several seeds.
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
from .monthly_activity_rnn import build_samples

torch.set_num_threads(1)
SEED = 42


def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """(batch, L) -> (batch, n_patches, patch_len) overlapping patches."""
    return x.unfold(dimension=1, size=patch_len, step=stride)


class PatchTST(nn.Module):
    def __init__(self, lookback: int, patch_len: int, stride: int, d_model: int, heads: int, layers: int, dropout: float, extra: int):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patches = (lookback - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(n_patches * d_model + extra, 1)

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        tokens = self.embed(patchify(x, self.patch_len, self.stride)) + self.position
        flat = self.encoder(tokens).flatten(1)
        if extra is not None:
            flat = torch.cat([flat, extra], dim=1)
        return self.head(flat)


def _fit_predict(series: pd.Series, target_month: pd.Timestamp, params: dict[str, object]) -> float:
    target_pos = series.index.get_loc(target_month)
    lookback, covariates = int(params["lookback"]), bool(params["covariates"])
    train_x, _, _, train_y, target_x, extra_train, extra_target = build_samples(series, target_pos, lookback, covariates)
    history = series.to_numpy(dtype=np.float32)[:target_pos]
    mean, scale = float(history.mean()), float(history.std()) or 1.0
    x_train = torch.from_numpy(((train_x - mean) / scale).astype(np.float32))
    x_target = torch.from_numpy(((target_x - mean) / scale).astype(np.float32))
    y_train = torch.from_numpy(((train_y - mean) / scale).reshape(-1, 1).astype(np.float32))
    e_train = torch.from_numpy(extra_train) if covariates else None
    e_target = torch.from_numpy(extra_target) if covariates else None
    outputs = []
    for seed in range(int(params["n_seeds"])):
        torch.manual_seed(int(params["seed"]) + seed)
        model = PatchTST(lookback, int(params["patch_len"]), int(params["stride"]), int(params["d_model"]), int(params["heads"]), int(params["layers"]), float(params["dropout"]), 2 if covariates else 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(int(params["epochs"])):
            optimizer.zero_grad()
            loss_fn(model(x_train, e_train), y_train).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            outputs.append(float(model(x_target, e_target).item()) * scale + mean)
    return float(np.mean(outputs))


def expanding_forecast(series: pd.Series, months: pd.DatetimeIndex, params: dict[str, object]) -> pd.Series:
    return pd.Series([_fit_predict(series, m, params) for m in months], index=months, name="forecast")


def candidates(n_trials: int, rng_seed: int = SEED) -> list[dict[str, object]]:
    rng = np.random.default_rng(rng_seed)
    space = {
        "lookback": (24, 36),
        "patch_len": (4, 6),
        "stride": (2, 3),
        "d_model": (16, 32),
        "heads": (2, 4),
        "layers": (1, 2),
        "dropout": (0.1, 0.2),
        "learning_rate": (0.002, 0.005),
        "weight_decay": (1e-4, 1e-2),
        "epochs": (40, 80),
        "covariates": (False, True),
    }
    out, seen = [], set()
    while len(out) < n_trials:
        pick = {key: values[int(rng.integers(len(values)))] for key, values in space.items()}
        key = tuple(sorted(pick.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append({**pick, "seed": SEED, "n_seeds": 1})
    return out


def run_experiment(output_dir: str | Path, n_trials: int = 10, final_seeds: int = 3) -> dict[str, object]:
    output = Path(output_dir)
    series = load_monthly_series(output)
    months = validation_months(series)
    actual = series.loc[months].to_numpy()
    trials = []
    search_path = output / "patchtst_search.csv"
    if search_path.exists():  # resume an interrupted search (candidates are deterministic)
        trials = pd.read_csv(search_path).to_dict("records")
    for i, params in enumerate(candidates(n_trials)):
        if i < len(trials):
            continue
        prediction = expanding_forecast(series, months, params).to_numpy()
        trials.append({**params, "validation_mae": float(np.mean(np.abs(prediction - actual))), "validation_rmse": float(np.sqrt(np.mean((prediction - actual) ** 2)))})
        pd.DataFrame(trials).to_csv(output / "patchtst_search.csv", index=False)
        print(f"  trial {i + 1}/{n_trials} val MAE={trials[-1]['validation_mae']:.3f}", flush=True)
    best = min(trials, key=lambda r: (r["validation_mae"], r["validation_rmse"]))
    params = {**{k: best[k] for k in trials[0] if not k.startswith("validation_")}, "n_seeds": final_seeds}
    frame = split_frame(series, expanding_forecast(series, forecast_months(series), params), "patchtst")
    rows = write_model_outputs(output, series, frame, "patchtst", {"architecture": "PatchTST-style (patch embedding + Transformer encoder, univariate)", "selected_hyperparameters": params, "search_trials": len(trials), "selection_metric": "validation MAE"})
    png_forecast_plot(output / "patchtst_forecast_vs_actual.png", frame, "PatchTST one-step forecast vs actual", "#b07aa1")
    return {"model": "patchtst", "selected_hyperparameters": params, "metrics": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--final-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output_dir, args.trials, args.final_seeds), indent=2, default=str))


if __name__ == "__main__":
    main()
