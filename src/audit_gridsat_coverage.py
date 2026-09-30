"""Sample GridSat crop coverage without creating a satellite training dataset."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from src.build_gridsat_dataset import HALF_CELLS, load_field
from src.gridsat_poc import source_url
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map

YEARS = (1980, 1983, 1986, 1990, 1995, 2000, 2005, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)


def sample_origins(path: Path) -> pd.DataFrame:
    states = filter_north_atlantic(load_ibtracs(path))
    splits = storm_split_map(states, 2015, 2019)
    meta = build_forecast_samples(states, 6).metadata
    rows = []
    for year in YEARS:
        candidates = meta.loc[meta.SEASON.eq(year)].sort_values("issue_time")
        if candidates.empty:
            raise ValueError(f"no eligible +6 h origin in {year}")
        row = candidates.iloc[len(candidates) // 2][["SID", "SEASON", "issue_time", "issue_lat", "issue_lon"]].copy()
        row["split"] = splits.loc[row.SID]
        rows.append(row)
    return pd.DataFrame(rows)


def valid_fraction(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, row: pd.Series) -> float:
    i = int(np.abs(lat - row.issue_lat).argmin())
    j = int(np.abs(((lon - row.issue_lon + 180) % 360) - 180).argmin())
    crop = field[i-HALF_CELLS:i+HALF_CELLS+1, j-HALF_CELLS:j+HALF_CELLS+1]
    if crop.shape != (201, 201):
        raise ValueError(f"unexpected crop dimensions {crop.shape}")
    return float(np.isfinite(crop).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ibtracs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for number, (_, row) in enumerate(sample_origins(args.ibtracs).iterrows(), 1):
        stamp = pd.Timestamp(row.issue_time)
        temporary = None
        print(f"[{number}/{len(YEARS)}] {stamp} {row.SID}", flush=True)
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
                temporary = Path(handle.name)
                with urlopen(source_url(stamp), timeout=600) as response:
                    while block := response.read(1024 * 1024): handle.write(block)
            field, lat, lon = load_field(temporary, stamp)
            records.append({**row.to_dict(), "valid_fraction": valid_fraction(field, lat, lon, row), "source_url": source_url(stamp), "download_bytes": temporary.stat().st_size})
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "coverage.csv", index=False)
    summary = {"sampled_origins": len(frame), "zero_fraction": float((frame.valid_fraction == 0).mean()), "below_0_5_fraction": float((frame.valid_fraction < .5).mean()), "at_least_0_9_fraction": float((frame.valid_fraction >= .9).mean()), "min": float(frame.valid_fraction.min()), "median": float(frame.valid_fraction.median()), "mean": float(frame.valid_fraction.mean()), "max": float(frame.valid_fraction.max()), "by_split": frame.groupby("split").valid_fraction.agg(["count", "min", "median", "mean", "max"]).to_dict(orient="index"), "by_year": frame.groupby("SEASON").valid_fraction.first().to_dict()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__": main()
