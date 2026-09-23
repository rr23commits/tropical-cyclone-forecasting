#!/usr/bin/env python3
"""Write a compact provenance manifest for the frozen track-only evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from .gru import GRUSettings


OUTPUTS = (
    "results/track_only_baselines.csv", "results/track_only_gru_comparison.csv",
    "results/track_only_gru_seed_results.csv", "results/track_only_gru_seed_summary.csv",
    "results/track_only_error_analysis/held_out_errors.csv",
    "results/track_only_evaluation/model_horizon_summary.csv",
    "results/track_only_evaluation/paired_gru_vs_ridge.csv",
    "results/track_only_eda/data_counts.csv",
)


def sha256(path: Path) -> str:
    """Hash an immutable input or generated output without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Record the code revision when Git metadata is available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest(source: Path, outputs: tuple[Path, ...]) -> dict[str, object]:
    """Return all fixed protocol values needed to identify this run."""
    settings = GRUSettings()
    return {
        "dataset": {"name": "IBTrACS v04r01 North Atlantic / hurdat_atl", "path": str(source), "sha256": sha256(source)},
        "code_commit": git_commit(),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "torch")},
        },
        "protocol": {
            "train_end_year": 2015, "validation_end_year": 2019, "test_years": "2020-2025",
            "history_times": ["t-24", "t-18", "t-12", "t-6", "t"], "horizons_hours": [6, 12, 24, 48],
            "features": ["latitude", "longitude", "wind", "north_step_km", "east_step_km", "speed_kmh", "direction_sin", "direction_cos", "wind_change"],
            "targets": ["north_displacement_km", "east_displacement_km", "wind_change_kt"],
            "gru": {"hidden_sizes": list(settings.hidden_sizes), "learning_rate": settings.learning_rate,
                    "weight_decay": settings.weight_decay, "batch_size": settings.batch_size,
                    "max_epochs": settings.max_epochs, "patience": settings.patience, "robustness_seeds": [42, 43, 44]},
        },
        "output_sha256": {str(path): sha256(path) for path in outputs if path.exists()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="official IBTrACS source used for this run")
    parser.add_argument("--output", type=Path, default=Path("results/track_only_provenance.json"))
    args = parser.parse_args()
    payload = manifest(args.csv, tuple(Path(path) for path in OUTPUTS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved provenance manifest: {args.output}")


if __name__ == "__main__":
    main()
