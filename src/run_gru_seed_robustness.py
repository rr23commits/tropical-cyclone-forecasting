#!/usr/bin/env python3
"""Repeat the frozen GRU protocol across fixed seeds; never select on test data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.evaluation import evaluate, split_samples
from src.gru import GRUSettings, predict_gru, train_select_gru
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map


SEEDS = (42, 43, 44)


def summarize_seed_results(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every fixed seed equally; no result is chosen after testing."""
    metrics = ["track_error_km", "wind_mae", "wind_rmse"]
    summary = rows.groupby("horizon_hours", as_index=False).agg(
        seeds=("seed", "nunique"), origins=("origins", "first"), storms=("storms", "first"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, "std") for metric in metrics},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--output", type=Path, default=Path("results/track_only_gru_seed_results.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/track_only_gru_seed_summary.csv"))
    args = parser.parse_args()
    data = filter_north_atlantic(load_ibtracs(args.csv))
    split = storm_split_map(data, 2015, 2019)
    rows: list[dict[str, object]] = []
    for horizon in (6, 12, 24, 48):
        groups = split_samples(build_forecast_samples(data, horizon), split)
        for seed in SEEDS:
            # Candidate width and early stopping remain validation-only for every seed.
            model = train_select_gru(groups["train"], groups["validation"], GRUSettings(seed=seed))
            metrics = evaluate(groups["test"], predict_gru(model, groups["test"]))
            rows.append({"horizon_hours": horizon, "seed": seed, **metrics,
                         "gru_hidden_size": model.hidden_size, "gru_epochs_trained": model.epochs_trained,
                         "gru_validation_mse": model.validation_mse, "torch_version": torch.__version__})
            print(f"+{horizon}h seed={seed}: track={metrics['track_error_km']:.2f} km wind-MAE={metrics['wind_mae']:.2f} kt")
    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summarize_seed_results(results).to_csv(args.summary_output, index=False)
    print(f"Saved fixed-seed GRU results: {args.output} and {args.summary_output}")


if __name__ == "__main__":
    main()
