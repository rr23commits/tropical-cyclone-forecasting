"""Build and validate a CDS request manifest for the frozen TCC cohort.

This module is metadata-only. It never creates a CDS client or submits requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.acquire_era5 import _canonical_payload, _payload_fingerprint
from src.preflight_genesis import GRID_DEGREES, PATCH_DEGREES, cohort


PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
SINGLE_DATASET = "reanalysis-era5-single-levels"
PRESSURE_VARIABLES = [
    "u_component_of_wind", "v_component_of_wind", "temperature",
    "specific_humidity", "vertical_velocity",
]
PRESSURE_LEVELS = ["200", "500", "700", "850"]
SST_VARIABLES = ["sea_surface_temperature"]
PRODUCTS = {
    "pressure": (PRESSURE_DATASET, PRESSURE_VARIABLES),
    "sst": (SINGLE_DATASET, SST_VARIABLES),
}
REQUIRED_PAIRS = {
    "u_component_of_wind": [200, 850],
    "v_component_of_wind": [200, 850],
    "temperature": [700],
    "specific_humidity": [700],
    "vertical_velocity": [500],
}
BYTES_PER_VALUE = 4


def _stamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def _position_rows(manifest: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    rows = cohort(manifest)
    if rows.duplicated(["tcc_track_id", "issue_time_utc"]).any():
        raise ValueError("the accepted TCC cohort contains duplicate track/issue rows")
    required = {"tcc_track_id", "time_utc", "tcc_lat", "tcc_lon"}
    if required - set(positions):
        raise ValueError(f"positions lack {sorted(required - set(positions))}")

    required_positions = {}
    for row in rows.itertuples(index=False):
        stamps = [_stamp(stamp) for stamp in row.input_times_utc.split("|")]
        if len(stamps) != 9 or stamps[-1] != _stamp(row.issue_time_utc):
            raise ValueError(f"TCC track {row.tcc_track_id} does not retain its nine issue-ending inputs")
        if any(right - left != pd.Timedelta(hours=3) for left, right in zip(stamps, stamps[1:])):
            raise ValueError(f"TCC track {row.tcc_track_id} input times are not frozen 3-hour steps")
        for stamp in stamps:
            required_positions[(str(row.tcc_track_id), stamp)] = None

    position_rows = positions.loc[:, ["tcc_track_id", "time_utc", "tcc_lat", "tcc_lon"]].copy()
    position_rows["tcc_track_id"] = position_rows.tcc_track_id.astype(str)
    position_rows["time_utc"] = pd.to_datetime(position_rows.time_utc, utc=True)
    if position_rows.duplicated(["tcc_track_id", "time_utc"]).any():
        raise ValueError("duplicate TCC track/time position; deduplicate before manifest generation")
    position_rows["tcc_lat"] = pd.to_numeric(position_rows.tcc_lat, errors="raise")
    position_rows["tcc_lon"] = pd.to_numeric(position_rows.tcc_lon, errors="raise")
    if not position_rows.tcc_lat.between(-90, 90).all() or not position_rows.tcc_lon.between(-180, 180).all():
        raise ValueError("TCC position coordinates are outside valid latitude/longitude bounds")

    actual = {(row.tcc_track_id, row.time_utc) for row in position_rows.itertuples(index=False)}
    missing, extra = set(required_positions) - actual, actual - set(required_positions)
    if missing or extra:
        raise ValueError(f"TCC position keys differ from frozen history: missing={len(missing)}, extra={len(extra)}")

    indexed = position_rows.set_index(["tcc_track_id", "time_utc"])
    for row in rows.itertuples(index=False):
        point = indexed.loc[(str(row.tcc_track_id), _stamp(row.issue_time_utc))]
        if not (math.isclose(point.tcc_lat, float(row.current_lat), abs_tol=1e-6)
                and math.isclose(point.tcc_lon, float(row.current_lon), abs_tol=1e-6)):
            raise ValueError(f"current TCC position disagrees at {row.tcc_track_id}/{row.issue_time_utc}")
    return position_rows.sort_values(["tcc_track_id", "time_utc"]).reset_index(drop=True)


def _cohort_fingerprint(manifest: pd.DataFrame) -> str:
    columns = ["tcc_track_id", "basin", "year", "issue_time_utc", "current_lat", "current_lon",
               "label_24h", "label_48h", "input_times_utc", "split"]
    missing = set(columns) - set(manifest)
    if missing:
        raise ValueError(f"cohort manifest lacks frozen fields: {sorted(missing)}")
    rows = cohort(manifest).loc[:, columns].copy()
    rows["tcc_track_id"] = rows.tcc_track_id.astype(str)
    records = sorted(
        [{column: "" if pd.isna(value) else str(value) for column, value in row.items()}
         for row in rows.to_dict(orient="records")],
        key=lambda row: (row["tcc_track_id"], row["issue_time_utc"]),
    )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _longitude_arc(longitudes: list[float]) -> tuple[float, float]:
    """Return the smallest unwrapped longitude interval containing all points."""
    values = sorted(value % 360 for value in longitudes)
    if len(values) == 1:
        return values[0], values[0]
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    gaps.append(values[0] + 360 - values[-1])
    gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    start_index = (gap_index + 1) % len(values)
    start = values[start_index]
    end = values[gap_index]
    if end < start:
        end += 360
    return start, end


def _signed_longitude(value: float) -> float:
    normalized = (value + 180) % 360 - 180
    return 0.0 if abs(normalized) < 1e-10 else round(normalized, 8)


def _area(group: pd.DataFrame) -> tuple[list[float], int, int]:
    pad = PATCH_DEGREES
    north = math.ceil((group.tcc_lat.max() + pad) / GRID_DEGREES) * GRID_DEGREES
    south = math.floor((group.tcc_lat.min() - pad) / GRID_DEGREES) * GRID_DEGREES
    west_unwrapped, east_unwrapped = _longitude_arc(group.tcc_lon.tolist())
    west_unwrapped = math.floor((west_unwrapped - pad) / GRID_DEGREES) * GRID_DEGREES
    east_unwrapped = math.ceil((east_unwrapped + pad) / GRID_DEGREES) * GRID_DEGREES
    if north > 90 or south < -90:
        raise ValueError("a 5-degree crop around a TCC position would cross a geographic pole")
    width = east_unwrapped - west_unwrapped
    height = north - south
    if width >= 30 or height >= 30:
        raise ValueError(f"TCC daily request is not regional: {height:g}x{width:g} degrees")
    columns = round(width / GRID_DEGREES) + 1
    rows = round(height / GRID_DEGREES) + 1
    area = [north, _signed_longitude(west_unwrapped), south, _signed_longitude(east_unwrapped)]
    return area, rows, columns


def _payload(dataset: str, variables: list[str], when: pd.Timestamp, hours: list[str], area: list[float],
             levels: list[str] | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {
        "product_type": ["reanalysis"], "variable": variables,
        "year": [when.strftime("%Y")], "month": [when.strftime("%m")],
        "day": [when.strftime("%d")], "time": hours,
        "area": area, "data_format": "netcdf", "download_format": "unarchived",
    }
    if levels is not None:
        request["pressure_level"] = levels
    return _canonical_payload(request)


def build_manifest(manifest: pd.DataFrame, positions: pd.DataFrame) -> dict[str, Any]:
    """Create one pressure and one SST request per deduplicated track × UTC day."""
    points = _position_rows(manifest, positions)
    request_rows = []
    request_cell_time = Counter()
    timestamps = set(points.time_utc)
    for (track_id, day), group in points.groupby(
        ["tcc_track_id", points.time_utc.dt.strftime("%Y-%m-%d")], sort=True
    ):
        area, lat_count, lon_count = _area(group)
        grid_cells = lat_count * lon_count
        times = sorted(group.time_utc.dt.strftime("%H:00").unique().tolist())
        day_stamp = pd.Timestamp(day, tz="UTC")
        batch_id = f"tcc-{track_id}-{day_stamp:%Y%m%d}"
        for product, (dataset, variables) in PRODUCTS.items():
            payload = _payload(
                dataset, variables, day_stamp, times, area,
                PRESSURE_LEVELS if product == "pressure" else None,
            )
            key = f"genesis_{product}_{track_id}_{day_stamp:%Y%m%d}"
            request_rows.append({
                "request_key": key,
                "batch_key": batch_id,
                "tcc_track_id": track_id,
                "utc_day": day_stamp.strftime("%Y-%m-%d"),
                "product": product,
                "dataset": dataset,
                "payload": payload,
                "payload_sha256": _payload_fingerprint(payload),
                "position_count": len(group),
                "crop_patch_degrees": [2 * PATCH_DEGREES, 2 * PATCH_DEGREES],
                "area_grid_cells": grid_cells,
                "area_latitude_points": lat_count,
                "area_longitude_points": lon_count,
            })
            request_cell_time[product] += grid_cells * len(times)

    pressure_pairs = sum(len(levels) for levels in REQUIRED_PAIRS.values())
    requested_pressure_pairs = len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS)
    pressure_values = request_cell_time["pressure"] * requested_pressure_pairs
    sst_values = request_cell_time["sst"]
    required_values = request_cell_time["pressure"] * pressure_pairs + sst_values
    crop_values = len(points) * 21 * 21 * (pressure_pairs + 1)
    unique_times = sorted(timestamps)
    counts = Counter(row["product"] for row in request_rows)
    areas = [row["payload"]["area"] for row in request_rows if row["product"] == "pressure"]
    area_heights = [north - south for north, _, south, _ in areas]
    area_widths = [(east - west) % 360 for _, west, _, east in areas]
    batches = len(areas)
    crop_centers = []
    for row in points.itertuples(index=False):
        grid_lat = round(row.tcc_lat / GRID_DEGREES) * GRID_DEGREES
        grid_lon = round(row.tcc_lon / GRID_DEGREES) * GRID_DEGREES
        crop_centers.append({
            "tcc_track_id": row.tcc_track_id,
            "time_utc": _iso(row.time_utc),
            "latitude": float(row.tcc_lat),
            "longitude": float(row.tcc_lon),
            "grid_center_latitude": round(grid_lat, 8),
            "grid_center_longitude": _signed_longitude(grid_lon),
            "crop_area_north_west_south_east": [
                round(grid_lat + PATCH_DEGREES, 8),
                _signed_longitude(grid_lon - PATCH_DEGREES),
                round(grid_lat - PATCH_DEGREES, 8),
                _signed_longitude(grid_lon + PATCH_DEGREES),
            ],
        })
    return {
        "schema_version": 1,
        "cohort": {
            "source_manifest_rows": len(manifest), "rows": 1276,
            "tcc_tracks": int(points.tcc_track_id.nunique()),
            "input_references_before_deduplication": 1276 * 9,
            "deduplicated_track_time_positions": len(points),
            "overlapping_references_removed": 1276 * 9 - len(points),
            "unique_utc_timestamps": len(set(unique_times)),
            "timestamp_start": _iso(unique_times[0]), "timestamp_end": _iso(unique_times[-1]),
            "hours": sorted({stamp.strftime("%H:00") for stamp in unique_times}),
            "label_split_fields": ["label_24h", "label_48h", "split"],
            "labels_and_splits_modified": False,
            "frozen_rows_sha256": _cohort_fingerprint(manifest),
        },
        "predictors": {
            "vorticity_850": "derive from u_component_of_wind and v_component_of_wind at 850 hPa",
            "wind_shear_850_200": "u_component_of_wind and v_component_of_wind at 850 and 200 hPa",
            "relative_humidity_700": "derive from temperature and specific_humidity at 700 hPa",
            "vertical_motion_500": "vertical_velocity at 500 hPa",
            "sea_surface_temperature": "sea_surface_temperature at surface",
        },
        "products": {
            "pressure": {
                "dataset": PRESSURE_DATASET,
                "variables": PRESSURE_VARIABLES,
                "levels_hpa": [int(level) for level in PRESSURE_LEVELS],
                "required_variable_level_pairs": REQUIRED_PAIRS,
                "request_variable_level_cartesian_count": requested_pressure_pairs,
                "required_pressure_pair_count": pressure_pairs,
                "extra_cartesian_pairs_requested": requested_pressure_pairs - pressure_pairs,
                "note": "CDS variable/level selection is Cartesian; extra combinations are explicitly counted.",
            },
            "sst": {"dataset": SINGLE_DATASET, "variables": SST_VARIABLES, "level": "surface"},
        },
        "geometry": {
            "grid_degrees": GRID_DEGREES,
            "patch_degrees": [2 * PATCH_DEGREES, 2 * PATCH_DEGREES],
            "patch_grid_cells_per_position": 21 * 21,
            "area_construction": "per track×UTC day; union of centered 5° patches; bounds rounded outward to 0.25°",
            "longitude_wrap": "smallest circular interval; CDS API west>east used across the dateline",
            "track_day_batches": batches,
            "distinct_dynamic_areas": len({tuple(area) for area in areas}),
            "dateline_crossing_areas": sum(area[1] > area[3] for area in areas),
            "latitude_envelope": [min(area[2] for area in areas), max(area[0] for area in areas)],
            "area_height_degrees_min_max": [min(area_heights), max(area_heights)],
            "area_width_degrees_min_max": [min(area_widths), max(area_widths)],
            "sample_areas_north_west_south_east": areas[:5],
        },
        "estimates": {
            "request_counts": {**dict(counts), "total": len(request_rows)},
            "requested_grid_cell_times": dict(request_cell_time),
            "pressure_requested_values": pressure_values,
            "required_predictor_values_before_crop": required_values,
            "deduplicated_crop_values": crop_values,
            "uncompressed_float32_request_bytes": (pressure_values + sst_values) * BYTES_PER_VALUE,
            "uncompressed_float32_required_pairs_bytes": required_values * BYTES_PER_VALUE,
            "uncompressed_float32_cropped_predictor_bytes": crop_values * BYTES_PER_VALUE,
            "uncompressed_float32_request_mib": round((pressure_values + sst_values) * BYTES_PER_VALUE / 2**20, 2),
            "uncompressed_float32_cropped_predictor_mib": round(crop_values * BYTES_PER_VALUE / 2**20, 2),
            "data_volume_note": "Float32 uncompressed value-array estimate; excludes NetCDF metadata/compression and counts Cartesian pressure combinations.",
        },
        "requests": request_rows,
        "crop_centers": crop_centers,
    }


def _iso(stamp: pd.Timestamp) -> str:
    return stamp.isoformat().replace("+00:00", "Z")


def validate_manifest(document: dict[str, Any], manifest: pd.DataFrame, positions: pd.DataFrame) -> None:
    """Regenerate and compare all payloads; fail closed on drift from frozen inputs."""
    expected = build_manifest(manifest, positions)
    if document != expected:
        raise ValueError("request manifest does not match the frozen cohort, positions, or canonical payloads")
    for request in document["requests"]:
        if request["payload_sha256"] != _payload_fingerprint(request["payload"]):
            raise ValueError(f"request payload fingerprint mismatch: {request['request_key']}")
        payload = request["payload"]
        expected_dataset, expected_variables = PRODUCTS[request["product"]]
        if request["dataset"] != expected_dataset or payload["variable"] != expected_variables:
            raise ValueError(f"unexpected ERA5 product or variable substitution: {request['request_key']}")
        if request["product"] == "pressure" and payload["pressure_level"] != PRESSURE_LEVELS:
            raise ValueError(f"unexpected pressure levels: {request['request_key']}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory)
            except OSError:
                pass
        finally:
            os.close(directory)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("positions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/genesis_era5_request_manifest.json"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cohort_manifest, positions = pd.read_csv(args.manifest), pd.read_csv(args.positions)
    if args.validate_only:
        document = json.loads(args.output.read_text())
        validate_manifest(document, cohort_manifest, positions)
    else:
        document = build_manifest(cohort_manifest, positions)
        validate_manifest(document, cohort_manifest, positions)
        _atomic_json(args.output, document)
    summary = {key: value for key, value in document.items() if key not in {"requests", "crop_centers"}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
