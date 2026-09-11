#!/usr/bin/env python3
"""Render a compact SVG comparison from the reproducible track-only result CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {"persistence": "#9ca3af", "constant-motion": "#f59e0b", "ridge": "#2563eb", "gru": "#059669"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="?", default=Path("results/phase3_gru_comparison.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/track_only_track_error.svg"))
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    horizons = sorted({int(row["horizon_hours"]) for row in rows})
    models = [name for name in COLORS if any(row["model"] == name for row in rows)]
    values = {(row["model"], int(row["horizon_hours"])): float(row["track_error_km"]) for row in rows}
    maximum = max(values.values())
    width, height, left, bottom, top = 760, 430, 80, 60, 45
    plot_width, plot_height = width - left - 25, height - bottom - top

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="80" y="25" font-family="sans-serif" font-size="18">North Atlantic track-only test error</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-25}" y2="{height-bottom}" stroke="#111"/>']
    for fraction in range(5):
        value = maximum * fraction / 4
        y = height - bottom - plot_height * fraction / 4
        lines += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-25}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                  f'<text x="8" y="{y+4:.1f}" font-family="sans-serif" font-size="11">{value:.0f} km</text>']
    for index, horizon in enumerate(horizons):
        x = left + plot_width * index / (len(horizons) - 1)
        lines.append(f'<text x="{x-12:.1f}" y="{height-30}" font-family="sans-serif" font-size="12">+{horizon}h</text>')
    for model_index, model in enumerate(models):
        points = []
        for index, horizon in enumerate(horizons):
            x = left + plot_width * index / (len(horizons) - 1)
            y = height - bottom - plot_height * values[(model, horizon)] / maximum
            points.append((x, y))
        lines.append(f'<polyline fill="none" stroke="{COLORS[model]}" stroke-width="2.5" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{COLORS[model]}"/>')
        lines.append(f'<rect x="{left + model_index * 145}" y="{height-16}" width="10" height="10" fill="{COLORS[model]}"/>')
        lines.append(f'<text x="{left + 15 + model_index * 145}" y="{height-7}" font-family="sans-serif" font-size="11">{model}</text>')
    lines.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
