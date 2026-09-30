"""Build resumable, issue-time GridSat crops for the frozen forecast protocol."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from src.gridsat_poc import _to_utc, source_name, source_url
from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs, storm_split_map

HORIZONS = (6, 12, 24, 48)
HALF_CELLS = 100
FIELD_SIZE = 201
MANIFEST_FIELDS = ("SID", "season", "split", "issue_time", "issue_lat", "issue_lon", "horizons", "crop_path", "height", "width", "valid_fraction", "source_file", "source_url", "source_time_utc", "download_bytes", "saved_crop_bytes", "decode")


def origins(path: Path) -> pd.DataFrame:
    """Reuse the frozen per-horizon builders, retaining no target values."""
    states = filter_north_atlantic(load_ibtracs(path))
    split_map = storm_split_map(states, 2015, 2019)
    rows = []
    for horizon in HORIZONS:
        meta = build_forecast_samples(states, horizon).metadata
        rows.append(pd.DataFrame({"SID": meta.SID, "season": meta.SEASON, "issue_time": meta.issue_time,
                                  "issue_lat": meta.issue_lat, "issue_lon": meta.issue_lon,
                                  "split": meta.SID.map(split_map), "horizon": horizon}))
    all_rows = pd.concat(rows, ignore_index=True)
    return all_rows.groupby(["SID", "issue_time"], as_index=False).agg(
        season=("season", "first"), issue_lat=("issue_lat", "first"), issue_lon=("issue_lon", "first"),
        split=("split", "first"), horizons=("horizon", lambda x: "|".join(map(str, sorted(x))))
    ).sort_values(["issue_time", "SID"]).reset_index(drop=True)


def crop_name(row: pd.Series) -> str:
    return f"{row.split}/{row.SID}_{pd.Timestamp(row.issue_time):%Y%m%dT%H%MZ}.npz"


def load_field(path: Path, expected_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as data:
        ir = data.variables["irwin_cdr"]
        ir.set_auto_maskandscale(False)
        raw = np.asarray(ir[:]).squeeze()
        missing = np.isin(raw, [getattr(ir, "_FillValue", -31999), getattr(ir, "missing_value", -31999)])
        field = raw.astype("float32") * float(ir.scale_factor) + float(ir.add_offset)
        field[missing | (field < 140.0) | (field > 375.0)] = np.nan
        time = _to_utc(data.variables["time"][0], data.variables["time"].units, getattr(data.variables["time"], "calendar", None))
        if time != expected_time:
            raise ValueError(f"GridSat timestamp mismatch: {time} != {expected_time}")
        return field, np.asarray(data.variables["lat"][:]), np.asarray(data.variables["lon"][:])


def write_crop(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, row: pd.Series, path: Path) -> dict[str, object]:
    i = int(np.abs(lat - row.issue_lat).argmin())
    j = int(np.abs(((lon - row.issue_lon + 180) % 360) - 180).argmin())
    if field.ndim != 2 or i < HALF_CELLS or j < HALF_CELLS or i + HALF_CELLS >= len(lat) or j + HALF_CELLS >= len(lon):
        raise ValueError(f"invalid crop geometry for {row.SID} {row.issue_time}")
    crop = field[i-HALF_CELLS:i+HALF_CELLS+1, j-HALF_CELLS:j+HALF_CELLS+1]
    if crop.shape != (FIELD_SIZE, FIELD_SIZE):
        raise ValueError(f"unexpected crop dimensions {crop.shape}")
    valid = np.isfinite(crop)
    if not valid.any():
        raise ValueError("crop is entirely invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, image=crop.astype("float16"))
    os.replace(temporary, path)
    return {"height": FIELD_SIZE, "width": FIELD_SIZE, "valid_fraction": float(valid.mean()), "saved_crop_bytes": path.stat().st_size}


def completed(manifest: Path) -> set[tuple[str, str]]:
    if not manifest.exists(): return set()
    frame = pd.read_csv(manifest, usecols=["SID", "issue_time"])
    return set(zip(frame.SID, frame.issue_time))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ibtracs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-source-files", type=int, default=100, help="0 processes all remaining source timestamps")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path, origins_path, failures_path = args.output / "manifest.csv", args.output / "origins.csv", args.output / "FAILED.json"
    fixed = origins(args.ibtracs)
    if origins_path.exists():
        prior = pd.read_csv(origins_path)
        if len(prior) != len(fixed): raise ValueError("existing origins.csv does not match frozen protocol")
    else: fixed.to_csv(origins_path, index=False)
    done = completed(manifest_path)
    pending = fixed.loc[~fixed.apply(lambda r: (r.SID, r.issue_time.isoformat()) in done, axis=1)].copy()
    groups = list(pending.groupby("issue_time", sort=True))
    if args.max_source_files: groups = groups[:args.max_source_files]
    write_header = not manifest_path.exists()
    downloaded = retained = success = 0
    try:
        for number, (stamp, rows) in enumerate(groups, 1):
            stamp = pd.Timestamp(stamp); source = None
            print(f"[{number}/{len(groups)}] {stamp} ({len(rows)} crops)", flush=True)
            try:
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
                    source = Path(handle.name)
                    with urlopen(source_url(stamp), timeout=600) as response:
                        while block := response.read(1024 * 1024): handle.write(block)
                size = source.stat().st_size; downloaded += size
                field, lat, lon = load_field(source, stamp)
                records = []
                for _, row in rows.iterrows():
                    path = args.output / "crops" / crop_name(row)
                    stats = write_crop(field, lat, lon, row, path); retained += stats["saved_crop_bytes"]; success += 1
                    records.append({**row.to_dict(), "issue_time": stamp.isoformat(), "crop_path": str(path.relative_to(args.output)),
                                    "source_file": source_name(stamp), "source_url": source_url(stamp), "source_time_utc": stamp.isoformat(),
                                    "download_bytes": size, "decode": "int16*0.01+200 K; fill only; physical 140-375 K", **stats})
                with manifest_path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS); writer.writeheader() if write_header else None; writer.writerows(records)
                write_header = False
            finally:
                if source is not None: source.unlink(missing_ok=True)
    except Exception as exc:
        failures_path.write_text(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n")
        raise
    frame = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    summary = {"eligible_origins": len(fixed), "completed_origins": len(frame), "remaining_origins": len(fixed)-len(frame), "this_run_success": success,
               "this_run_download_bytes": downloaded, "this_run_retained_bytes": retained,
               "minimum_valid_fraction": None if frame.empty else float(frame.valid_fraction.min()), "median_valid_fraction": None if frame.empty else float(frame.valid_fraction.median()),
               "counts_by_split": fixed.groupby("split").size().to_dict()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
