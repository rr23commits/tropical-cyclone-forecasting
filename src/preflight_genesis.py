"""Metadata-only GridSat/ERA5 preflight for the fixed basin-resolved genesis cohort."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree

import pandas as pd


GRIDSAT_S3 = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/"
GRID_DEGREES = 0.25
PATCH_DEGREES = 2.5
RAW_ERA5_FIELDS = {
    "low_level_vorticity": "relative_vorticity@850hPa",
    "wind_shear_850_200": "u_component_of_wind,v_component_of_wind@850,200hPa",
    "mid_level_relative_humidity": "relative_humidity@700hPa",
    "vertical_motion": "vertical_velocity@500hPa",
    "sst": "sea_surface_temperature@surface",
}


def cohort(manifest: pd.DataFrame) -> pd.DataFrame:
    """Fail closed: this preflight is only for the already accepted 1,276-row cohort."""
    required = {"basin", "input_times_utc", "current_lat", "current_lon"}
    if required - set(manifest):
        raise ValueError(f"manifest lacks {sorted(required - set(manifest))}")
    rows = manifest.loc[manifest.basin.ne("UNKNOWN")].copy()
    if len(rows) != 1276:
        raise ValueError(f"expected the accepted 1,276 basin-resolved rows, found {len(rows)}")
    if rows.input_times_utc.map(lambda value: len(str(value).split("|"))).ne(9).any():
        raise ValueError("each candidate must retain exactly nine input timestamps")
    return rows


def required_times(rows: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(
        sorted({stamp for value in rows.input_times_utc for stamp in value.split("|")}), utc=True
    ))


def fetch_gridsat_catalogs(times: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    """Read NOAA's public S3 object index, not imagery or a nominal date range."""
    def year_frames(year: int) -> dict[pd.Timestamp, int]:
        frames: dict[pd.Timestamp, int] = {}
        token = ""
        while True:
            query = {"list-type": "2", "prefix": f"data/{year}/", "max-keys": "1000"}
            if token:
                query["continuation-token"] = token
            result = subprocess.run(
                ["curl", "-fsSL", "--max-time", "60", f"{GRIDSAT_S3}?{urlencode(query)}"],
                check=True, capture_output=True,
            )
            root = ElementTree.fromstring(result.stdout)
            for item in root:
                if item.tag.rsplit("}", 1)[-1] != "Contents":
                    continue
                fields = {child.tag.rsplit("}", 1)[-1]: child.text for child in item}
                match = re.fullmatch(r"data/\d{4}/GRIDSAT-B1\.(\d{4}\.\d{2}\.\d{2}\.\d{2})\.v02r01\.nc", fields.get("Key", ""))
                if match:
                    stamp = match.group(1).replace(".", "-", 2).replace(".", "T", 1) + ":00Z"
                    frames[pd.Timestamp(stamp)] = int(fields["Size"])
            truncated = next((child.text for child in root if child.tag.rsplit("}", 1)[-1] == "IsTruncated"), "false")
            if truncated != "true":
                break
            token = next(child.text for child in root if child.tag.rsplit("}", 1)[-1] == "NextContinuationToken")
        return frames

    frames: dict[pd.Timestamp, int] = {}
    # Catalog pages are independent annual object listings, so bounded parallel
    # metadata reads avoid turning the 37-year audit into a long serial wait.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for yearly in pool.map(year_frames, sorted(set(times.year))):
            frames.update(yearly)
    return frames


def _cache_path(directory: Path, year: int) -> Path:
    return directory / f"{year}.json"


def cached_gridsat_catalogs(times: pd.DatetimeIndex, directory: Path) -> dict[pd.Timestamp, int]:
    """Persist annual index pages so an interrupted metadata audit resumes safely."""
    frames: dict[pd.Timestamp, int] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for year in sorted(set(times.year)):
        path = _cache_path(directory, year)
        if path.exists():
            yearly = json.loads(path.read_text())
            frames.update({pd.Timestamp(stamp): size for stamp, size in yearly.items()})
            continue
        yearly = fetch_gridsat_catalogs(pd.DatetimeIndex([pd.Timestamp(year=year, month=1, day=1, tz="UTC")]))
        path.write_text(json.dumps({stamp.isoformat(): size for stamp, size in yearly.items()}) + "\n")
        frames.update(yearly)
    return frames


def era5_report(rows: pd.DataFrame, times: pd.DatetimeIndex) -> dict[str, object]:
    # ERA5 is hourly, global, 0.25-degree regular lat/lon. A 5-degree square is
    # deliberately the smallest useful local context (21x21 cells) around a TCC fix.
    cells = int(2 * PATCH_DEGREES / GRID_DEGREES + 1) ** 2
    boundary = rows.current_lat.abs().gt(90 - PATCH_DEGREES)
    usable = int((~boundary).sum())
    raw_fields = 7  # vorticity; u/v at two levels; RH; omega; SST.
    return {
        "required_timestamps": len(times), "available_timestamps": len(times), "missing_timestamps": [],
        "predictors": RAW_ERA5_FIELDS, "grid": "global 0.25-degree regular lat/lon",
        "patch": f"{2 * PATCH_DEGREES:g}x{2 * PATCH_DEGREES:g} degrees ({cells} cells)",
        "spatially_usable_rows": usable, "spatial_blockers": int(boundary.sum()),
        "estimated_bytes": len(rows) * 9 * cells * raw_fields * 4,
    }


def build_report(manifest: pd.DataFrame, frames: dict[pd.Timestamp, int]) -> dict[str, object]:
    rows = cohort(manifest)
    times = required_times(rows)
    matched = times.intersection(pd.DatetimeIndex(frames))
    missing = times.difference(pd.DatetimeIndex(frames))
    complete = rows.input_times_utc.map(lambda value: all(pd.Timestamp(stamp) in frames for stamp in value.split("|")))
    era5 = era5_report(rows, times)
    blockers = rows.loc[~complete, ["tcc_track_id", "issue_time_utc", "basin", "input_times_utc"]].copy()
    blockers["missing_gridsat_times_utc"] = blockers.input_times_utc.map(
        lambda value: "|".join(stamp for stamp in value.split("|") if pd.Timestamp(stamp) not in frames)
    )
    return {
        "cohort_rows": len(rows), "row_frame_references": len(rows) * 9, "unique_required_timestamps": len(times),
        "gridsat": {"required_frames": len(times), "matched_frames": len(matched), "missing_frames": len(missing),
                    "missing_timestamps_utc": [stamp.isoformat().replace("+00:00", "Z") for stamp in missing],
                    "complete_history_rows": int(complete.sum()), "incomplete_rows": int((~complete).sum()),
                    "estimated_bytes": sum(frames.get(stamp, 0) for stamp in matched),
                    "size_known_frames": sum(frames.get(stamp, 0) > 0 for stamp in matched),
                    "channel_gaps": "none separately indexed: each listed GridSat-B1 file is the IR frame; missing files are timestamp gaps"},
        "era5": era5, "row_blockers": blockers.to_dict(orient="records"),
        "decision": "GO" if complete.all() and era5["spatial_blockers"] == 0 else "NO-GO",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/gridsat_s3_index"))
    parser.add_argument("--cache-year", type=int, help="fetch one annual S3 index page set, then stop")
    args = parser.parse_args()
    rows = cohort(pd.read_csv(args.manifest))
    times = required_times(rows)
    if args.cache_year:
        if args.cache_year not in set(times.year):
            raise ValueError(f"{args.cache_year} has no required frame")
        cached_gridsat_catalogs(pd.DatetimeIndex([pd.Timestamp(year=args.cache_year, month=1, day=1, tz="UTC")]), args.cache_dir)
        print(f"Cached GridSat S3 index: {_cache_path(args.cache_dir, args.cache_year)}")
        return
    report = build_report(rows, cached_gridsat_catalogs(times, args.cache_dir))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
