#!/usr/bin/env python3
"""Create a compact report from archived held-out predictions; never train a model."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .error_analysis import summarize


MODELS = ("persistence", "constant-motion", "ridge", "gru")
COLORS = {"persistence": "#6b7280", "constant-motion": "#d97706", "ridge": "#2563eb", "gru": "#059669"}


def paired_gru_ridge(rows: pd.DataFrame, bootstrap_samples: int = 2000) -> pd.DataFrame:
    """Compare GRU and Ridge only at their shared held-out origins, clustered by storm."""
    keys = ["SID", "issue_time", "target_time", "horizon_hours"]
    columns = keys + ["track_error_km", "wind_abs_error_kt"]
    gru = rows.loc[rows.model.eq("gru"), columns].rename(columns={
        "track_error_km": "gru_track_error_km", "wind_abs_error_kt": "gru_wind_abs_error_kt",
    })
    ridge = rows.loc[rows.model.eq("ridge"), columns].rename(columns={
        "track_error_km": "ridge_track_error_km", "wind_abs_error_kt": "ridge_wind_abs_error_kt",
    })
    paired = gru.merge(ridge, on=keys, validate="one_to_one")
    paired["track_delta_gru_minus_ridge_km"] = paired.gru_track_error_km - paired.ridge_track_error_km
    paired["wind_mae_delta_gru_minus_ridge_kt"] = paired.gru_wind_abs_error_kt - paired.ridge_wind_abs_error_kt
    output: list[dict[str, float | int]] = []
    random = np.random.default_rng(42)
    for horizon, group in paired.groupby("horizon_hours", sort=True):
        storms = group.groupby("SID", observed=True).mean(numeric_only=True)
        indices = random.integers(0, len(storms), size=(bootstrap_samples, len(storms)))
        track_draws = storms.track_delta_gru_minus_ridge_km.to_numpy()[indices].mean(axis=1)
        wind_draws = storms.wind_mae_delta_gru_minus_ridge_kt.to_numpy()[indices].mean(axis=1)
        # The bootstrap samples whole-storm means, so its point estimate must
        # use the same equal-storm weighting rather than the origin-weighted
        # mean printed in the primary model table.
        output.append({
            "horizon_hours": int(horizon), "common_origins": int(len(group)), "common_storms": int(len(storms)),
            "storm_macro_gru_track_mean_km": float(storms.gru_track_error_km.mean()), "storm_macro_ridge_track_mean_km": float(storms.ridge_track_error_km.mean()),
            "storm_macro_track_delta_gru_minus_ridge_km": float(storms.track_delta_gru_minus_ridge_km.mean()),
            "storm_track_delta_ci95_low_km": float(np.quantile(track_draws, .025)), "storm_track_delta_ci95_high_km": float(np.quantile(track_draws, .975)),
            "storm_macro_gru_wind_mae_kt": float(storms.gru_wind_abs_error_kt.mean()), "storm_macro_ridge_wind_mae_kt": float(storms.ridge_wind_abs_error_kt.mean()),
            "storm_macro_wind_mae_delta_gru_minus_ridge_kt": float(storms.wind_mae_delta_gru_minus_ridge_kt.mean()),
            "storm_wind_mae_delta_ci95_low_kt": float(np.quantile(wind_draws, .025)), "storm_wind_mae_delta_ci95_high_kt": float(np.quantile(wind_draws, .975)),
        })
    return pd.DataFrame(output)


def line_plot(table: pd.DataFrame, value: str, title: str, output: Path, x: str = "horizon_hours") -> None:
    """Render a small dependency-free line plot suitable for the report/viva."""
    width, height, left, bottom, top = 760, 380, 80, 55, 45
    plot_width, plot_height = width - left - 30, height - bottom - top
    x_values = sorted(table[x].unique())
    maximum = max(float(table[value].max()), 1.0)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-30}" y2="{height-bottom}" stroke="#111"/>']
    for fraction in range(5):
        y = height - bottom - plot_height * fraction / 4
        lines += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-30}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                  f'<text x="8" y="{y+4:.1f}" font-family="sans-serif" font-size="11">{maximum * fraction / 4:.1f}</text>']
    for index, value_x in enumerate(x_values):
        x_coord = left + plot_width * index / max(1, len(x_values) - 1)
        label = f'+{value_x}h' if x == "horizon_hours" else str(value_x)
        lines.append(f'<text x="{x_coord-12:.1f}" y="{height-28}" font-family="sans-serif" font-size="11">{label}</text>')
    for index, (model, group) in enumerate(table.groupby("model", sort=False)):
        group = group.set_index(x).loc[x_values]
        points = []
        for position, row in enumerate(group.itertuples()):
            x_coord = left + plot_width * position / max(1, len(x_values) - 1)
            y_coord = height - bottom - plot_height * float(getattr(row, value)) / maximum
            points.append((x_coord, y_coord))
        color = COLORS.get(model, "#111827")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in points) + '"/>')
        lines.append(f'<text x="{left + index * 160}" y="{height-8}" font-family="sans-serif" font-size="11" fill="{color}">{html.escape(model)}</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def signed_line_plot(table: pd.DataFrame, value: str, title: str, output: Path, x: str = "horizon_hours", group: str = "model") -> None:
    """Render signed residual means with a visible zero reference line."""
    width, height, left, bottom, top = 760, 380, 80, 55, 45
    plot_width, plot_height = width - left - 30, height - bottom - top
    x_values = list(dict.fromkeys(table[x].tolist()))
    maximum = max(abs(float(table[value].min())), abs(float(table[value].max())), 1.0)
    def y_coord(value_y: float) -> float:
        return height - bottom - plot_height * (value_y + maximum) / (2 * maximum)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-30}" y2="{height-bottom}" stroke="#111"/>',
             f'<line x1="{left}" y1="{y_coord(0):.1f}" x2="{width-30}" y2="{y_coord(0):.1f}" stroke="#111" stroke-dasharray="4 4"/>']
    for fraction in range(5):
        value_y = -maximum + 2 * maximum * fraction / 4
        y = y_coord(value_y)
        lines += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-30}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                  f'<text x="8" y="{y+4:.1f}" font-family="sans-serif" font-size="11">{value_y:.1f}</text>']
    for index, value_x in enumerate(x_values):
        x_coord = left + plot_width * index / max(1, len(x_values) - 1)
        label = f'+{value_x}h' if x == "horizon_hours" else str(value_x)
        lines.append(f'<text x="{x_coord-18:.1f}" y="{height-28}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
    palette = {**COLORS, 6: "#2563eb", 12: "#7c3aed", 24: "#d97706", 48: "#dc2626"}
    for index, (label, values) in enumerate(table.groupby(group, sort=False)):
        values = values.set_index(x).reindex(x_values)
        points = []
        for position, row in enumerate(values.itertuples()):
            x_coord = left + plot_width * position / max(1, len(x_values) - 1)
            points.append((x_coord, y_coord(float(getattr(row, value)))))
        color = palette.get(label, "#111827")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in points) + '"/>')
        lines.append(f'<text x="{left + index * 150}" y="{height-8}" font-family="sans-serif" font-size="11" fill="{color}">{html.escape(str(label))}</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def residual_summary(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Summarize signed held-out residuals; this is descriptive, not a new test."""
    return rows.groupby(by, observed=True, dropna=False).agg(
        origins=("SID", "size"), storms=("SID", "nunique"),
        track_mean_km=("track_error_km", "mean"),
        north_bias_km=("north_error_km", "mean"), east_bias_km=("east_error_km", "mean"),
        north_mae_km=("north_error_km", lambda values: values.abs().mean()),
        east_mae_km=("east_error_km", lambda values: values.abs().mean()),
        wind_bias_kt=("wind_error_kt", "mean"), wind_mae_kt=("wind_abs_error_kt", "mean"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/track_only_error_analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/track_only_evaluation"))
    args = parser.parse_args()
    rows = pd.read_csv(args.input_dir / "held_out_errors.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, ["model", "horizon_hours"])
    summary.to_csv(args.output_dir / "model_horizon_summary.csv", index=False)
    paired_gru_ridge(rows).to_csv(args.output_dir / "paired_gru_vs_ridge.csv", index=False)
    residual_summary(rows, ["model", "horizon_hours"]).to_csv(args.output_dir / "residual_by_horizon.csv", index=False)
    residual_summary(rows, ["model", "horizon_hours", "issue_intensity_bin"]).to_csv(args.output_dir / "residual_by_intensity.csv", index=False)
    residual_summary(rows, ["model", "horizon_hours", "motion_bin"]).to_csv(args.output_dir / "residual_by_motion.csv", index=False)
    difficulty = []
    for filename, category in (("by_motion.csv", "motion"), ("by_intensity.csv", "issue_intensity"), ("by_wind_change.csv", "intensity_change")):
        table = pd.read_csv(args.input_dir / filename)
        difficulty.append(table.assign(difficulty=category))
    pd.concat(difficulty, ignore_index=True).to_csv(args.output_dir / "difficulty_summary.csv", index=False)
    pd.read_csv(args.input_dir / "by_year.csv").to_csv(args.output_dir / "yearly_stability.csv", index=False)
    line_plot(summary, "track_mean_km", "Held-out mean track error (km)", args.output_dir / "track_error_by_model.svg")
    line_plot(summary, "wind_mae_kt", "Held-out wind MAE (kt)", args.output_dir / "wind_mae_by_model.svg")
    yearly = pd.read_csv(args.input_dir / "by_year.csv").query("model in ['gru', 'ridge']")
    line_plot(yearly, "track_mean_km", "GRU and Ridge yearly track error (km)", args.output_dir / "yearly_gru_ridge_track.svg", x="SEASON")
    residuals = residual_summary(rows, ["model", "horizon_hours"])
    signed_line_plot(residuals, "wind_bias_kt", "Held-out wind bias (kt)", args.output_dir / "wind_bias_by_horizon.svg")
    signed_line_plot(residuals, "north_bias_km", "Held-out north displacement bias (km)", args.output_dir / "north_bias_by_horizon.svg")
    signed_line_plot(residuals, "east_bias_km", "Held-out east displacement bias (km)", args.output_dir / "east_bias_by_horizon.svg")
    gru_intensity = residual_summary(rows.loc[rows.model.eq("gru")], ["horizon_hours", "issue_intensity_bin"])
    signed_line_plot(gru_intensity, "wind_bias_kt", "GRU wind bias by issue intensity (kt)", args.output_dir / "gru_wind_bias_by_intensity.svg", x="issue_intensity_bin", group="horizon_hours")
    gru_motion = residual_summary(rows.loc[rows.model.eq("gru")], ["horizon_hours", "motion_bin"])
    signed_line_plot(gru_motion, "wind_bias_kt", "GRU wind bias by target motion (kt)", args.output_dir / "gru_wind_bias_by_motion.svg", x="motion_bin", group="horizon_hours")
    print(f"Saved evaluation tables and plots: {args.output_dir}")


if __name__ == "__main__":
    main()
