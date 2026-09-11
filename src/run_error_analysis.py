#!/usr/bin/env python3
"""Recreate the frozen experiment and export only its held-out error analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baselines import constant_motion, flattened_features, persistence, select_ridge
from src.error_analysis import error_rows, summarize
from src.evaluation import split_samples
from src.gru import GRUSettings, predict_gru, train_select_gru
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--train-end", type=int, default=2015)
    parser.add_argument("--validation-end", type=int, default=2019)
    parser.add_argument("--output-dir", type=Path, default=Path("results/track_only_error_analysis"))
    args = parser.parse_args()

    filtered = filter_north_atlantic(load_ibtracs(args.csv))
    split_map = storm_split_map(filtered, args.train_end, args.validation_end)
    all_rows: list[pd.DataFrame] = []
    settings = GRUSettings()
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
        all_rows.extend(error_rows(groups["test"], prediction, model) for model, prediction in predictions.items())
        print(f"Analysed +{horizon}h: {len(groups['test'].metadata)} held-out origins; GRU width={gru.hidden_size}")

    rows = pd.concat(all_rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "held_out_errors.csv", index=False)
    summarize(rows, ["model", "horizon_hours"]).to_csv(args.output_dir / "overall.csv", index=False)
    summarize(rows, ["model", "horizon_hours", "issue_intensity_bin"]).to_csv(args.output_dir / "by_intensity.csv", index=False)
    summarize(rows, ["model", "horizon_hours", "motion_bin"]).to_csv(args.output_dir / "by_motion.csv", index=False)
    summarize(rows, ["model", "horizon_hours", "wind_change_bin"]).to_csv(args.output_dir / "by_wind_change.csv", index=False)
    summarize(rows, ["model", "horizon_hours", "SEASON"]).to_csv(args.output_dir / "by_year.csv", index=False)
    storm = summarize(rows.loc[rows["model"].eq("gru")], ["horizon_hours", "SID", "SEASON"])
    storm.sort_values(["horizon_hours", "track_mean_km"], ascending=[True, False]).to_csv(args.output_dir / "gru_by_storm.csv", index=False)
    print(f"Saved held-out errors and five summaries: {args.output_dir}")


if __name__ == "__main__":
    main()
