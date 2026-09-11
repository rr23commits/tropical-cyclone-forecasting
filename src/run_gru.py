#!/usr/bin/env python3
"""Run the compact GRU and the fixed Phase 2 baselines on identical test origins."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.baselines import constant_motion, flattened_features, persistence, select_ridge
from src.evaluation import evaluate, split_samples
from src.gru import GRUSettings, predict_gru, train_select_gru
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--train-end", type=int, default=2015)
    parser.add_argument("--validation-end", type=int, default=2019)
    parser.add_argument("--output", type=Path, default=Path("results/phase3_gru_comparison.csv"))
    args = parser.parse_args()

    settings = GRUSettings()
    filtered = filter_north_atlantic(load_ibtracs(args.csv))
    split_map = storm_split_map(filtered, args.train_end, args.validation_end)
    results: list[dict[str, float | int | str]] = []
    print(f"Split seasons: train <= {args.train_end}, validation <= {args.validation_end}, test later")
    print("horizon  model             origins  storms  track-km  wind-mae  wind-rmse  selection")

    for horizon in (6, 12, 24, 48):
        groups = split_samples(build_forecast_samples(filtered, horizon), split_map)
        ridge = select_ridge(groups["train"], groups["validation"])
        gru = train_select_gru(groups["train"], groups["validation"], settings)
        predictions = {
            "persistence": persistence(groups["test"]),
            "constant-motion": constant_motion(groups["test"]),
            "ridge": ridge.predict(flattened_features(groups["test"])),
            "gru": predict_gru(gru, groups["test"]),
        }
        expected_rows = len(groups["test"].metadata)
        if any(len(prediction) != expected_rows for prediction in predictions.values()):
            raise RuntimeError("models did not produce predictions for identical test origins")
        for name, prediction in predictions.items():
            metrics = evaluate(groups["test"], prediction)
            selection = (
                f"width={gru.hidden_size},epoch={gru.epochs_trained}" if name == "gru"
                else f"alpha={ridge.alpha:g}" if name == "ridge" else "-"
            )
            print(
                f"+{horizon:<2}h     {name:<17} {metrics['origins']:>7} {metrics['storms']:>7} "
                f"{metrics['track_error_km']:>9.2f} {metrics['wind_mae']:>9.2f} "
                f"{metrics['wind_rmse']:>10.2f} {selection}"
            )
            results.append({
                "dataset": "IBTrACS v04r01 NA / hurdat_atl",
                "train_end_year": args.train_end,
                "validation_end_year": args.validation_end,
                "horizon_hours": horizon,
                "split": "test",
                "train_origins": len(groups["train"].metadata),
                "train_storms": int(groups["train"].metadata["SID"].nunique()),
                "validation_origins": len(groups["validation"].metadata),
                "validation_storms": int(groups["validation"].metadata["SID"].nunique()),
                "model": name,
                "origins": metrics["origins"],
                "storms": metrics["storms"],
                "track_error_km": metrics["track_error_km"],
                "wind_mae": metrics["wind_mae"],
                "wind_rmse": metrics["wind_rmse"],
                "ridge_alpha": ridge.alpha if name == "ridge" else "",
                "gru_hidden_size": gru.hidden_size if name == "gru" else "",
                "gru_epochs_trained": gru.epochs_trained if name == "gru" else "",
                "gru_validation_mse": gru.validation_mse if name == "gru" else "",
                "gru_seed": settings.seed if name == "gru" else "",
                "torch_version": torch.__version__,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
