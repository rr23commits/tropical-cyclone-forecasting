#!/usr/bin/env python3
"""Compare compact GRU forecasts across historical-window lengths."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.evaluation import align_common_origins, evaluate, split_samples
from src.gru import GRUSettings, predict_gru, train_select_gru
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map


HISTORY_STEPS = (2, 3, 5, 7, 9)


def history_hours(steps: int) -> int:
    """Convert state count to elapsed history; the issue state is included."""
    return (steps - 1) * 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--train-end", type=int, default=2015)
    parser.add_argument("--validation-end", type=int, default=2019)
    parser.add_argument("--output", type=Path, default=Path("results/phase4_history_ablation.csv"))
    args = parser.parse_args()

    settings = GRUSettings()
    filtered = filter_north_atlantic(load_ibtracs(args.csv))
    split_map = storm_split_map(filtered, args.train_end, args.validation_end)
    results: list[dict[str, float | int | str]] = []
    print(f"Split seasons: train <= {args.train_end}, validation <= {args.validation_end}, test later")
    print("horizon  history  split       origins  storms  track-km  wind-mae  wind-rmse  width/epoch")

    for horizon in (6, 24, 48):
        groups_by_steps = {
            steps: split_samples(build_forecast_samples(filtered, horizon, history_steps=steps), split_map)
            for steps in HISTORY_STEPS
        }
        aligned_validation = align_common_origins([groups_by_steps[steps]["validation"] for steps in HISTORY_STEPS])
        aligned_test = align_common_origins([groups_by_steps[steps]["test"] for steps in HISTORY_STEPS])
        validation_rows: list[dict[str, float | int | str]] = []

        for position, steps in enumerate(HISTORY_STEPS):
            gru = train_select_gru(groups_by_steps[steps]["train"], groups_by_steps[steps]["validation"], settings)
            for split_name, samples in (("validation_common", aligned_validation[position]), ("test_common", aligned_test[position])):
                metrics = evaluate(samples, predict_gru(gru, samples))
                row: dict[str, float | int | str] = {
                    "dataset": "IBTrACS v04r01 NA / hurdat_atl",
                    "train_end_year": args.train_end,
                    "validation_end_year": args.validation_end,
                    "horizon_hours": horizon,
                    "history_hours": history_hours(steps),
                    "history_states": steps,
                    "split": split_name,
                    "selected_by_validation_track": "",
                    "origins": metrics["origins"],
                    "storms": metrics["storms"],
                    "track_error_km": metrics["track_error_km"],
                    "wind_mae": metrics["wind_mae"],
                    "wind_rmse": metrics["wind_rmse"],
                    "gru_hidden_size": gru.hidden_size,
                    "gru_epochs_trained": gru.epochs_trained,
                    "gru_validation_mse": gru.validation_mse,
                    "gru_seed": settings.seed,
                    "torch_version": torch.__version__,
                }
                results.append(row)
                if split_name == "validation_common":
                    validation_rows.append(row)
                print(
                    f"+{horizon:<2}h     {history_hours(steps):>3}h     {split_name:<16} "
                    f"{metrics['origins']:>7} {metrics['storms']:>7} {metrics['track_error_km']:>9.2f} "
                    f"{metrics['wind_mae']:>9.2f} {metrics['wind_rmse']:>10.2f} "
                    f"{gru.hidden_size}/{gru.epochs_trained}"
                )

        best = min(validation_rows, key=lambda row: float(row["track_error_km"]))
        best_history = best["history_hours"]
        for row in results:
            if row["horizon_hours"] == horizon and row["history_hours"] == best_history:
                row["selected_by_validation_track"] = "yes"
        print(f"+{horizon:<2}h best validation history: {best_history}h")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
