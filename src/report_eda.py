#!/usr/bin/env python3
"""Create a deterministic, training-aware EDA report for the track-only study."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import split_samples
from .ibtracs import (
    CORE_STATE_COLUMNS, FEATURE_NAMES, build_forecast_samples, filter_north_atlantic,
    load_ibtracs, storm_split_map,
)


def quality_tables(raw: pd.DataFrame, filtered: pd.DataFrame, train_end: int, validation_end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return transparent sequential eligibility and missing-value counts."""
    timestamp = raw["ISO_TIME"]
    scope = raw.loc[
        raw["BASIN"].eq("NA") & raw["TRACK_TYPE"].eq("main") & raw["SEASON"].between(1980, 2025)
    ].copy()
    scoped_time = timestamp.loc[scope.index]
    six_hour = scope.loc[scoped_time.notna() & scoped_time.dt.minute.eq(0) & scoped_time.dt.hour.mod(6).eq(0)]
    source = six_hour.loc[six_hour["USA_AGENCY"].eq("hurdat_atl")]
    tropical = source.loc[source["USA_STATUS"].isin({"TD", "TS", "HU", "TY", "ST", "TC", "SD", "SS"})]
    complete = tropical.dropna(subset=CORE_STATE_COLUMNS)
    split = storm_split_map(filtered, train_end, validation_end)
    counts = pd.DataFrame([
        {"stage": "loaded records", "states": len(raw), "storms": raw.SID.nunique()},
        {"stage": "North Atlantic main records, 1980-2025", "states": len(scope), "storms": scope.SID.nunique()},
        {"stage": "exact six-hour ticks", "states": len(six_hour), "storms": six_hour.SID.nunique()},
        {"stage": "HURDAT Atlantic agency", "states": len(source), "storms": source.SID.nunique()},
        {"stage": "eligible tropical/subtropical statuses", "states": len(tropical), "storms": tropical.SID.nunique()},
        {"stage": "complete position and wind", "states": len(complete), "storms": complete.SID.nunique()},
        {"stage": "final eligible retained states", "states": len(filtered), "storms": filtered.SID.nunique()},
        *[{"stage": f"retained {label} storms", "states": int(filtered.SID.map(split).eq(label).sum()), "storms": int((split == label).sum())}
          for label in ("train", "validation", "test")],
    ])
    missing = six_hour.loc[:, [*CORE_STATE_COLUMNS, "USA_PRES"]].isna().mean().mul(100).rename_axis("field").reset_index(name="missing_percent")
    missing.insert(1, "six_hour_states_examined", len(six_hour))
    return counts, missing


def split_horizon_counts(data: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Count exact eligible forecast origins for the frozen protocol."""
    rows = []
    for horizon in (6, 12, 24, 48):
        for label, samples in split_samples(build_forecast_samples(data, horizon), split).items():
            rows.append({"horizon_hours": horizon, "split": label, "origins": len(samples.metadata), "storms": samples.metadata.SID.nunique()})
    return pd.DataFrame(rows)


def distribution_tables(data: pd.DataFrame, split: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe only training inputs/targets, avoiding test-informed EDA choices."""
    train = split_samples(build_forecast_samples(data, 6), split)["train"]
    features = pd.DataFrame(train.features.reshape(-1, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    feature_summary = features.describe(percentiles=[.05, .25, .5, .75, .95]).T.rename_axis("feature").reset_index()
    targets = []
    for horizon in (6, 12, 24, 48):
        sample = split_samples(build_forecast_samples(data, horizon), split)["train"]
        frame = pd.DataFrame(sample.targets, columns=["north_displacement_km", "east_displacement_km", "wind_change_kt"])
        summary = frame.describe(percentiles=[.05, .25, .5, .75, .95]).T.rename_axis("target").reset_index()
        summary.insert(0, "horizon_hours", horizon)
        targets.append(summary)
    return feature_summary, pd.concat(targets, ignore_index=True)


def _svg(title: str, width: int = 800, height: int = 420) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="45" y="26" font-family="sans-serif" font-size="18">{html.escape(title)}</text>']


def distribution_plot(features: pd.DataFrame, targets_24h: pd.DataFrame, output: Path) -> None:
    """Plot compact training-only histograms for representative inputs and labels."""
    panels = [(features["wind"], "Input wind (kt)"), (features["speed_kmh"], "Input speed (km/h)"),
              (targets_24h["wind_change_kt"], "+24h wind change (kt)"),
              (targets_24h["north_displacement_km"], "+24h north displacement (km)")]
    lines = _svg("Training-only feature and target distributions", 800, 460)
    for index, (values, title) in enumerate(panels):
        x0, y0 = 50 + (index % 2) * 380, 55 + (index // 2) * 200
        clean = values.dropna().to_numpy(float)
        counts, edges = np.histogram(clean, bins=16)
        maximum = max(counts.max(), 1)
        lines += [f'<text x="{x0}" y="{y0}" font-family="sans-serif" font-size="13">{html.escape(title)}</text>',
                  f'<line x1="{x0}" y1="{y0+145}" x2="{x0+320}" y2="{y0+145}" stroke="#111"/>']
        for bin_index, count in enumerate(counts):
            x = x0 + bin_index * 20
            height = 120 * count / maximum
            lines.append(f'<rect x="{x:.1f}" y="{y0+145-height:.1f}" width="18" height="{height:.1f}" fill="#2563eb"/>')
        lines += [f'<text x="{x0}" y="{y0+162}" font-family="sans-serif" font-size="10">{edges[0]:.1f}</text>',
                  f'<text x="{x0+285}" y="{y0+162}" font-family="sans-serif" font-size="10">{edges[-1]:.1f}</text>']
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def representative_storms(data: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Select fixed training-only quantiles by retained-state count for illustrations."""
    train = data.loc[data.SID.map(split).eq("train")]
    sizes = train.groupby("SID").size().sort_values()
    picks = [sizes.index[int((len(sizes) - 1) * q)] for q in (.25, .5, .9)]
    return train.loc[train.SID.isin(picks)].copy()


def trajectory_plot(storms: pd.DataFrame, output: Path) -> None:
    """Render selected training tracks using a shared geographic extent."""
    lines = _svg("Representative training cyclone trajectories", 800, 480)
    left, top, right, bottom = 70, 55, 760, 420
    lon_min, lon_max = storms.USA_LON.min(), storms.USA_LON.max()
    lat_min, lat_max = storms.USA_LAT.min(), storms.USA_LAT.max()
    lon_pad, lat_pad = max((lon_max - lon_min) * .08, 1), max((lat_max - lat_min) * .08, 1)
    lon_min, lon_max, lat_min, lat_max = lon_min-lon_pad, lon_max+lon_pad, lat_min-lat_pad, lat_max+lat_pad
    lines += [f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#f8fafc" stroke="#111"/>']
    colors = ("#2563eb", "#059669", "#d97706")
    for color, (sid, storm) in zip(colors, storms.groupby("SID", sort=True)):
        points = []
        for row in storm.sort_values("ISO_TIME").itertuples():
            x = left + (row.USA_LON - lon_min) / (lon_max - lon_min) * (right-left)
            y = bottom - (row.USA_LAT - lat_min) / (lat_max - lat_min) * (bottom-top)
            points.append((x, y))
        lines += [f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>',
                  f'<text x="{left}" y="{440 - colors.index(color)*16}" font-family="sans-serif" font-size="11" fill="{color}">{html.escape(str(sid))}</text>']
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def wind_plot(storms: pd.DataFrame, output: Path) -> None:
    """Render wind evolution for the same fixed training-only storms."""
    lines = _svg("Representative training cyclone wind histories", 800, 480)
    left, top, right, bottom = 70, 55, 760, 420
    maximum = max(float(storms.USA_WIND.max()), 1.0)
    lines += [f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#f8fafc" stroke="#111"/>']
    colors = ("#2563eb", "#059669", "#d97706")
    for color, (sid, storm) in zip(colors, storms.groupby("SID", sort=True)):
        ordered = storm.sort_values("ISO_TIME")
        points = [(left + i / max(len(ordered)-1, 1) * (right-left), bottom - row.USA_WIND / maximum * (bottom-top)) for i, row in enumerate(ordered.itertuples())]
        lines += [f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>',
                  f'<text x="{left}" y="{440 - colors.index(color)*16}" font-family="sans-serif" font-size="11" fill="{color}">{html.escape(str(sid))}</text>']
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results/track_only_eda"))
    parser.add_argument("--train-end", type=int, default=2015)
    parser.add_argument("--validation-end", type=int, default=2019)
    args = parser.parse_args()
    raw = load_ibtracs(args.csv)
    data = filter_north_atlantic(raw)
    split = storm_split_map(data, args.train_end, args.validation_end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts, missing = quality_tables(raw, data, args.train_end, args.validation_end)
    features, targets = distribution_tables(data, split)
    counts.to_csv(args.output_dir / "data_counts.csv", index=False)
    missing.to_csv(args.output_dir / "missingness.csv", index=False)
    split_horizon_counts(data, split).to_csv(args.output_dir / "split_horizon_counts.csv", index=False)
    features.to_csv(args.output_dir / "training_feature_distribution.csv", index=False)
    targets.to_csv(args.output_dir / "training_target_distribution.csv", index=False)
    # Use the original train feature matrix for the plot; keeping it local avoids a second data path.
    train = split_samples(build_forecast_samples(data, 6), split)["train"]
    train_24h = split_samples(build_forecast_samples(data, 24), split)["train"]
    target_values = pd.DataFrame(train_24h.targets, columns=["north_displacement_km", "east_displacement_km", "wind_change_kt"])
    distribution_plot(pd.DataFrame(train.features.reshape(-1, len(FEATURE_NAMES)), columns=FEATURE_NAMES), target_values, args.output_dir / "training_distributions.svg")
    storms = representative_storms(data, split)
    storms.loc[:, ["SID", "SEASON", "NAME", "ISO_TIME", "USA_LAT", "USA_LON", "USA_WIND"]].to_csv(args.output_dir / "representative_training_storms.csv", index=False)
    trajectory_plot(storms, args.output_dir / "representative_trajectories.svg")
    wind_plot(storms, args.output_dir / "representative_wind_histories.svg")
    (args.output_dir / "README.md").write_text(
        "# Track-only EDA\n\nAll feature and target distribution summaries use training origins only. "
        "The three displayed storms are deterministic training-only length quantiles; they are illustrations, not selected for model performance. "
        "`data_counts.csv` and `missingness.csv` document raw-data eligibility before the frozen forecasting split.\n",
        encoding="utf-8",
    )
    print(f"Saved EDA report: {args.output_dir}")


if __name__ == "__main__":
    main()
