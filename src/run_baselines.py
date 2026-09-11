#!/usr/bin/env python3
"""Run Phase 2 baselines on the audited IBTrACS North Atlantic subset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.baselines import constant_motion, flattened_features, persistence, select_ridge
from src.evaluation import evaluate, split_samples
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--train-end", type=int, default=2015)
    parser.add_argument("--validation-end", type=int, default=2019)
    parser.add_argument("--output", type=Path, default=Path("results/phase2_baselines.csv"))
    args = parser.parse_args()

    filtered = filter_north_atlantic(load_ibtracs(args.csv))
    splits = storm_split_map(filtered, args.train_end, args.validation_end)
    print(f"Split seasons: train <= {args.train_end}, validation <= {args.validation_end}, test later")
    print("horizon  model             origins  storms  track-km  wind-mae  wind-rmse  alpha")
    results: list[dict[str, float | int | str]] = []

    for horizon in (6, 12, 24, 48):
        samples = build_forecast_samples(filtered, horizon)
        groups = split_samples(samples, splits)
        ridge = select_ridge(groups["train"], groups["validation"])
        predictions = {
            "persistence": persistence(groups["test"]),
            "constant-motion": constant_motion(groups["test"]),
            "ridge": ridge.predict(flattened_features(groups["test"])),
        }
        for name, prediction in predictions.items():
            metrics = evaluate(groups["test"], prediction)
            alpha = f"{ridge.alpha:g}" if name == "ridge" else "-"
            print(
                f"+{horizon:<2}h     {name:<17} {metrics['origins']:>7} {metrics['storms']:>7} "
                f"{metrics['track_error_km']:>9.2f} {metrics['wind_mae']:>9.2f} "
                f"{metrics['wind_rmse']:>10.2f} {alpha:>6}"
            )
            results.append({
                "dataset": "IBTrACS v04r01 NA / hurdat_atl",
                "train_end_year": args.train_end,
                "validation_end_year": args.validation_end,
                "horizon_hours": horizon,
                "model": name,
                "ridge_alpha": alpha,
                **metrics,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
