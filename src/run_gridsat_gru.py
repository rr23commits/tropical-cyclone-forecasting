#!/usr/bin/env python3
"""Evaluate matched GridSat CNN+GRU and track-only GRU controls by horizon."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.evaluation import evaluate
from src.gridsat_gru import join_gridsat_samples, predict_gridsat_gru, split_gridsat_samples, train_select_gridsat_gru
from src.gru import GRUSettings, predict_gru, train_select_gru
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--gridsat-root", type=Path, required=True, help="external dataset directory containing manifest.csv and crops/")
    parser.add_argument("--train-end", type=int, default=2015)
    parser.add_argument("--validation-end", type=int, default=2019)
    parser.add_argument("--output", type=Path, default=Path("results/gridsat_gru_comparison.csv"))
    args = parser.parse_args()

    data = filter_north_atlantic(load_ibtracs(args.csv))
    split_map = storm_split_map(data, args.train_end, args.validation_end)
    settings = GRUSettings()
    results: list[dict[str, float | int | str]] = []
    print(f"Split seasons: train <= {args.train_end}, validation <= {args.validation_end}, test later")
    print("horizon  model                 origins  storms  track-km  wind-mae  wind-rmse  selection")
    for horizon in (6, 12, 24, 48):
        matched = join_gridsat_samples(build_forecast_samples(data, horizon), args.gridsat_root)
        groups = split_gridsat_samples(matched, split_map)
        if any(not len(groups[split].samples.metadata) for split in ("train", "validation", "test")):
            raise ValueError(f"GridSat-matched +{horizon} h origins must cover train, validation, and test")
        control = train_select_gru(groups["train"].samples, groups["validation"].samples, settings)
        multimodal = train_select_gridsat_gru(groups["train"], groups["validation"], settings)
        predictions = {
            "gridsat-subset-gru": predict_gru(control, groups["test"].samples),
            "gridsat-cnn-gru": predict_gridsat_gru(multimodal, groups["test"]),
        }
        for name, prediction in predictions.items():
            metrics = evaluate(groups["test"].samples, prediction)
            selection = (f"width={control.hidden_size},epoch={control.epochs_trained}" if name == "gridsat-subset-gru"
                         else f"width={multimodal.hidden_size},epoch={multimodal.epochs_trained}")
            print(f"+{horizon:<2}h     {name:<21} {metrics['origins']:>7} {metrics['storms']:>7} {metrics['track_error_km']:>9.2f} {metrics['wind_mae']:>9.2f} {metrics['wind_rmse']:>10.2f} {selection}")
            results.append({
                "dataset": "IBTrACS v04r01 NA / hurdat_atl + external GridSat-B1 IR",
                "train_end_year": args.train_end,
                "validation_end_year": args.validation_end, "horizon_hours": horizon, "split": "test",
                "train_origins": len(groups["train"].samples.metadata), "train_storms": int(groups["train"].samples.metadata.SID.nunique()),
                "validation_origins": len(groups["validation"].samples.metadata), "validation_storms": int(groups["validation"].samples.metadata.SID.nunique()),
                "model": name, **metrics,
                "gru_hidden_size": control.hidden_size if name == "gridsat-subset-gru" else multimodal.hidden_size,
                "epochs_trained": control.epochs_trained if name == "gridsat-subset-gru" else multimodal.epochs_trained,
                "validation_mse": control.validation_mse if name == "gridsat-subset-gru" else multimodal.validation_mse,
                "seed": settings.seed, "torch_version": torch.__version__,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
