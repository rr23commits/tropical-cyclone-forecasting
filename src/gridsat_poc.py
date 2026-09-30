"""Download 11 exact-time GridSat crops for the satellite feasibility POC."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs


S3 = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data"
HALF_WIDTH_DEGREES = 7.0
HORIZON_HOURS = 24
YEARS = {
    "train": (1981, 1995, 2005, 2015),
    "validation": (2016, 2018, 2019),
    "test": (2020, 2022, 2023, 2024),
}


def source_name(stamp: pd.Timestamp) -> str:
    return f"GRIDSAT-B1.{stamp:%Y.%m.%d.%H}.v02r01.nc"


def source_url(stamp: pd.Timestamp) -> str:
    return f"{S3}/{stamp:%Y}/{source_name(stamp)}"


def selected_origins(raw_path: Path) -> pd.DataFrame:
    """Select one mid-lifetime +24 h origin per fixed POC season.

    Target availability establishes that a row is a real forecasting origin; no
    target field is retained or sent to the satellite service.
    """
    states = filter_north_atlantic(load_ibtracs(raw_path))
    states = states.loc[states.SEASON.le(2024)].copy()
    origins = build_forecast_samples(states, HORIZON_HOURS).metadata
    rows: list[pd.Series] = []
    for split, years in YEARS.items():
        for year in years:
            candidates = origins.loc[origins.SEASON.eq(year)].sort_values("issue_time")
            if candidates.empty:
                raise ValueError(f"no +{HORIZON_HOURS} h origins for POC season {year}")
            row = candidates.iloc[len(candidates) // 2].copy()
            row["split"] = split
            rows.append(row)
    return pd.DataFrame(rows).loc[:, ["SID", "SEASON", "split", "issue_time", "issue_lat", "issue_lon"]]


def _to_utc(value: object, units: str, calendar: str | None) -> pd.Timestamp:
    converted = num2date(value, units=units, calendar=calendar or "standard", only_use_cftime_datetimes=False)
    return pd.Timestamp(converted).tz_localize("UTC")


def retrieve_crop(row: pd.Series, crop_path: Path) -> dict[str, object]:
    stamp = pd.Timestamp(row.issue_time)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    if stamp.minute or stamp.second or stamp.hour % 3:
        raise ValueError(f"issue time is not a GridSat timestamp: {stamp}")

    url = source_url(stamp)
    source_path: Path | None = None
    try:
        # GridSat is chunked as global files. Stream one file to temporary disk,
        # crop it, then always delete it before advancing to the next origin.
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
            source_path = Path(handle.name)
            with urlopen(url, timeout=600) as response:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        download_bytes = source_path.stat().st_size
        with Dataset(source_path) as data:
            full = np.ma.filled(data.variables["irwin_cdr"][:], np.nan).astype("float32").squeeze()
            lat = np.asarray(data.variables["lat"][:], dtype="float32")
            lon = np.asarray(data.variables["lon"][:], dtype="float32")
            time_var = data.variables["time"]
            actual_time = _to_utc(time_var[0], time_var.units, getattr(time_var, "calendar", None))
    finally:
        if source_path is not None:
            source_path.unlink(missing_ok=True)

    if actual_time != stamp:
        raise ValueError(f"GridSat response time {actual_time} does not match issue time {stamp}")
    if full.ndim != 2:
        raise ValueError(f"expected a 2-D GridSat IR field, got {full.shape}")
    lat_index = int(np.abs(lat - float(row.issue_lat)).argmin())
    lon_distance = np.abs(((lon - float(row.issue_lon) + 180) % 360) - 180)
    lon_index = int(lon_distance.argmin())
    half_cells = 100
    if lat_index < half_cells or lat_index + half_cells >= len(lat) or lon_index < half_cells or lon_index + half_cells >= len(lon):
        raise ValueError(f"201x201 crop exceeds GridSat grid boundary at {stamp}")
    values = full[lat_index - half_cells:lat_index + half_cells + 1, lon_index - half_cells:lon_index + half_cells + 1]
    crop_lat = lat[lat_index - half_cells:lat_index + half_cells + 1]
    crop_lon = lon[lon_index - half_cells:lon_index + half_cells + 1]
    nearest_lat = float(lat[lat_index])
    nearest_lon = float(lon[lon_index])
    valid = np.isfinite(values)
    np.savez_compressed(crop_path, irwin_cdr=values, lat=crop_lat, lon=crop_lon)
    return {
        "source_file": source_name(stamp),
        "source_time_utc": stamp.isoformat(),
        "returned_time_utc": actual_time.isoformat(),
        "source_url": url,
        "access_method": "public S3 source object; local 201x201 crop",
        "download_bytes": download_bytes,
        "saved_crop_bytes": crop_path.stat().st_size,
        "height": int(values.shape[0]),
        "width": int(values.shape[1]),
        "nearest_grid_lat": nearest_lat,
        "nearest_grid_lon": nearest_lon,
        "center_offset_lat_deg": nearest_lat - float(row.issue_lat),
        "center_offset_lon_deg": ((nearest_lon - float(row.issue_lon) + 180) % 360) - 180,
        "valid_fraction": float(valid.mean()),
        "missing_pixels": int((~valid).sum()),
        "valid_min_kelvin": float(np.nanmin(values)),
        "valid_max_kelvin": float(np.nanmax(values)),
        "valid_mean_kelvin": float(np.nanmean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibtracs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing POC output: {args.output}")
    args.output.mkdir(parents=True)
    crops = args.output / "crops"
    crops.mkdir()

    manifest = selected_origins(args.ibtracs)
    reports: list[dict[str, object]] = []
    for number, (_, row) in enumerate(manifest.iterrows(), start=1):
        crop_path = crops / f"{number:02d}_{row.SID}_{pd.Timestamp(row.issue_time):%Y%m%dT%H%MZ}.npz"
        print(f"[{number}/{len(manifest)}] {row.SID} {row.issue_time}", flush=True)
        try:
            report = retrieve_crop(row, crop_path)
        except Exception as exc:
            failure = {
                "failed_origin": row.to_dict(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (args.output / "FAILED.json").write_text(json.dumps(failure, indent=2, default=str) + "\n")
            raise RuntimeError(f"GridSat POC failed at {row.SID} {row.issue_time}; see FAILED.json") from exc
        report["crop_file"] = str(crop_path.relative_to(args.output))
        reports.append(report)
    report_frame = pd.DataFrame(reports)
    output = pd.concat([manifest.reset_index(drop=True), report_frame], axis=1)
    output.to_csv(args.output / "manifest.csv", index=False)
    summary = {
        "purpose": "Small GridSat-B1 exact-time crop POC; not a training dataset or model result.",
        "origins": len(output),
        "horizon_hours_used_only_to_define_origins": HORIZON_HOURS,
        "splits": output.split.value_counts().sort_index().to_dict(),
        "all_exact_timestamps": bool((output.source_time_utc == output.returned_time_utc).all()),
        "crop_shapes": sorted({f"{height}x{width}" for height, width in zip(output.height, output.width)}),
        "valid_fraction_min": float(output.valid_fraction.min()),
        "total_download_bytes": int(output.download_bytes.sum()),
        "total_saved_crop_bytes": int(output.saved_crop_bytes.sum()),
        "no_future_information": "Only issue_time and issue_lat/lon form the source selection/crop; targets are not retained in manifest or sent to NOAA.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
