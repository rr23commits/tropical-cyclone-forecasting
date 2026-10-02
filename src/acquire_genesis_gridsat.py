"""Resumable GridSat crop acquisition for the frozen Genesis/TCC cohort."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.manifest_genesis_era5 import _position_rows
from src.preflight_genesis import cohort


REQUEST_FIELDS = (
    "tcc_track_id", "time_utc", "latitude", "longitude", "split",
    "source_timestamp_utc", "source_file", "source_url", "crop_path",
)
ATTEMPT_FIELDS = REQUEST_FIELDS + (
    "status", "error", "height", "width", "valid_fraction", "download_bytes", "saved_crop_bytes",
)


def source_name(stamp: pd.Timestamp) -> str:
    """Reuse GridSat's canonical source-file naming without duplicating it."""
    from src.gridsat_poc import source_name as canonical_source_name
    return canonical_source_name(stamp)


def source_url(stamp: pd.Timestamp) -> str:
    """Reuse GridSat's canonical public-source URL convention."""
    from src.gridsat_poc import source_url as canonical_source_url
    return canonical_source_url(stamp)


def _crop_name(row: pd.Series) -> str:
    from src.build_gridsat_dataset import crop_name
    return crop_name(row)


def load_field(path: Path, expected_time: pd.Timestamp):
    from src.build_gridsat_dataset import load_field as canonical_load_field
    return canonical_load_field(path, expected_time)


def write_crop(field, lat, lon, row: pd.Series, path: Path):
    from src.build_gridsat_dataset import write_crop as canonical_write_crop
    return canonical_write_crop(field, lat, lon, row, path)


def request_records(manifest: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Derive one crop request per validated, deduplicated TCC track/time fix."""
    rows = cohort(manifest)
    splits = rows.assign(tcc_track_id=rows.tcc_track_id.astype(str)).groupby("tcc_track_id").split
    if splits.nunique().gt(1).any():
        raise ValueError("a TCC track must not cross frozen splits")
    points = _position_rows(manifest, positions).rename(columns={"tcc_lat": "latitude", "tcc_lon": "longitude"})
    points["tcc_track_id"] = points.tcc_track_id.astype(str)
    points["split"] = points.tcc_track_id.map(splits.first())
    points["source_timestamp_utc"] = pd.to_datetime(points.time_utc, utc=True)
    points["source_file"] = points.source_timestamp_utc.map(source_name)
    points["source_url"] = points.source_timestamp_utc.map(source_url)
    points["crop_path"] = points.apply(
        lambda row: _crop_name(pd.Series({"split": row["split"], "SID": f"tcc-{row['tcc_track_id']}", "issue_time": row["time_utc"]})),
        axis=1,
    )
    records = points.loc[:, ["tcc_track_id", "time_utc", "latitude", "longitude", "split", "source_timestamp_utc", "source_file", "source_url", "crop_path"]].copy()
    records["time_utc"] = records.time_utc.map(lambda stamp: stamp.isoformat())
    records["source_timestamp_utc"] = records.source_timestamp_utc.map(lambda stamp: stamp.isoformat())
    if records.duplicated(["tcc_track_id", "time_utc"]).any():
        raise ValueError("Genesis GridSat crop requests must be unique by TCC track/time")
    return records.sort_values(["source_timestamp_utc", "tcc_track_id"]).reset_index(drop=True)


def _write_requests(path: Path, records: pd.DataFrame) -> None:
    if path.exists():
        prior = pd.read_csv(path, dtype={"tcc_track_id": str})
        if not prior.loc[:, REQUEST_FIELDS].equals(records.loc[:, REQUEST_FIELDS]):
            raise ValueError("existing Genesis GridSat requests do not match the frozen cohort")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(path, index=False)


def _completed(manifest_path: Path, crop_root: Path) -> set[tuple[str, str]]:
    if not manifest_path.exists():
        return set()
    attempts = pd.read_csv(manifest_path, dtype={"tcc_track_id": str})
    done = set()
    for row in attempts.itertuples(index=False):
        key = (str(row.tcc_track_id), str(row.time_utc))
        if row.status == "excluded":
            done.add(key)
        elif row.status == "saved" and (crop_root / row.crop_path).exists():
            done.add(key)
    return done


def _append(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTEMPT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def acquire(records: pd.DataFrame, crop_root: Path, attempt_manifest: Path, max_source_files: int) -> dict[str, int]:
    """Download each pending source once, then crop every pending TCC position from it."""
    done = _completed(attempt_manifest, crop_root)
    pending = records.loc[~records.apply(lambda row: (str(row.tcc_track_id), str(row.time_utc)) in done, axis=1)]
    groups = list(pending.groupby("source_timestamp_utc", sort=True))
    if max_source_files:
        groups = groups[:max_source_files]
    summary = {"unique_source_timestamps": len(groups), "saved": 0, "excluded": 0, "failed": 0}
    for number, (value, group) in enumerate(groups, 1):
        stamp = pd.Timestamp(value)
        source: Path | None = None
        print(f"[{number}/{len(groups)}] {stamp} ({len(group)} crops)", flush=True)
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
                source = Path(handle.name)
                with urlopen(source_url(stamp), timeout=600) as response:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
            download_bytes = source.stat().st_size
            field, lat, lon = load_field(source, stamp)
        except Exception as error:
            summary["failed"] += len(group)
            _append(attempt_manifest, [{
                **{name: getattr(row, name) for name in REQUEST_FIELDS}, "status": "failed",
                "error": f"{type(error).__name__}: {error}", "height": 0, "width": 0,
                "valid_fraction": "", "download_bytes": 0, "saved_crop_bytes": 0,
            } for row in group.itertuples(index=False)])
        else:
            for row in group.itertuples(index=False):
                try:
                    path = crop_root / row.crop_path
                    stats = write_crop(field, lat, lon, pd.Series({"issue_lat": row.latitude, "issue_lon": row.longitude}), path)
                    status = str(stats["status"])
                    summary[status] += 1
                    _append(attempt_manifest, [{
                        **{name: getattr(row, name) for name in REQUEST_FIELDS}, "status": status,
                        "error": str(stats.get("exclusion_reason", "")), "height": stats["height"],
                        "width": stats["width"], "valid_fraction": stats["valid_fraction"],
                        "download_bytes": download_bytes, "saved_crop_bytes": stats["saved_crop_bytes"],
                    }])
                except Exception as error:
                    summary["failed"] += 1
                    _append(attempt_manifest, [{
                        **{name: getattr(row, name) for name in REQUEST_FIELDS}, "status": "failed",
                        "error": f"{type(error).__name__}: {error}", "height": 0, "width": 0,
                        "valid_fraction": "", "download_bytes": download_bytes, "saved_crop_bytes": 0,
                    }])
        finally:
            if source is not None:
                source.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/genesis_candidate_manifest.csv"))
    parser.add_argument("--positions", type=Path, default=Path("results/lps_tcc_crosswalk/tcc_positions.csv"))
    parser.add_argument("--state-dir", type=Path, default=Path("results/genesis_gridsat"), help="small resumable manifests only")
    parser.add_argument("--crop-root", type=Path, required=True, help="external directory for extracted NPZ crops")
    parser.add_argument("--max-source-files", type=int, default=100, help="0 processes every pending source timestamp")
    args = parser.parse_args()
    if args.max_source_files < 0:
        parser.error("--max-source-files must be non-negative")
    records = request_records(pd.read_csv(args.manifest), pd.read_csv(args.positions))
    requests_path = args.state_dir / "genesis_gridsat_requests.csv"
    attempts_path = args.state_dir / "genesis_gridsat_acquisition_manifest.csv"
    _write_requests(requests_path, records)
    summary = acquire(records, args.crop_root, attempts_path, args.max_source_files)
    summary.update({"crop_requests": len(records), "all_unique_source_timestamps": int(records.source_timestamp_utc.nunique())})
    args.state_dir.mkdir(parents=True, exist_ok=True)
    (args.state_dir / "genesis_gridsat_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
