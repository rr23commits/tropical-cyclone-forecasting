#!/usr/bin/env python3
"""Acquire, validate, and extract the fixed Phase 4 ERA5 inputs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs


AREA = [55, -110, 5, -5]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
SELECTED_HISTORIES = ((6, 9), (24, 7), (48, 9))
ERA5_MONTHS = (9, 10)
ERA5_MANIFEST_NAME = "request_manifest_sep_oct.csv"
CDS_CLIENT_OPTIONS = {"timeout": 30, "retry_max": 1, "sleep_max": 5}
# A cyclic/broken pagination link must not turn resume into an unbounded read.
AMBIGUOUS_HISTORY_MAX_PAGES = 100
DOWNLOAD_ATTEMPTS = 3
ACQUISITION_CONCURRENCY = 4
ERA5_STATE_DIR = Path(__file__).resolve().parents[1] / "data/raw/era5/v1"
PRODUCTS = {
    "pressure_wind": {
        "dataset": "reanalysis-era5-pressure-levels",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": ["200", "500", "700", "850"],
        "data_variables": {"u", "v"},
        "units": {"u": "m s-1", "v": "m s-1"},
    },
    "humidity_700": {
        "dataset": "reanalysis-era5-pressure-levels",
        "variable": ["relative_humidity"],
        "pressure_level": ["700"],
        "data_variables": {"r"},
        "units": {"r": "%"},
    },
    "sst": {
        "dataset": "reanalysis-era5-single-levels",
        "variable": ["sea_surface_temperature"],
        "data_variables": {"sst"},
        "units": {"sst": "K"},
    },
}


@dataclass(frozen=True)
class FileSpec:
    product: str
    year: int
    month: int
    days: tuple[str, ...]
    request_key: str | None = None
    request_dataset: str | None = None
    request_payload: dict[str, Any] | None = None

    @property
    def month_key(self) -> str:
        return f"{self.year:04d}{self.month:02d}"

    @property
    def filename(self) -> str:
        if self.request_key:
            return f"{self.request_key}.nc"
        return f"{self.product}_{self.month_key}.nc"


GENESIS_PRODUCTS = {
    "pressure": {
        "dataset": "reanalysis-era5-pressure-levels",
        "variable": ["u_component_of_wind", "v_component_of_wind", "temperature",
                     "specific_humidity", "vertical_velocity"],
        "pressure_level": ["200", "500", "700", "850"],
        "data_variables": {"u", "v", "t", "q", "w"},
        "units": {"u": "m s-1", "v": "m s-1", "t": "K", "q": "kg kg-1", "w": "Pa s-1"},
    },
    "sst": {
        "dataset": "reanalysis-era5-single-levels",
        "variable": ["sea_surface_temperature"],
        "data_variables": {"sst"},
        "units": {"sst": "K"},
    },
}


def selected_origins(csv_path: Path) -> pd.DataFrame:
    """Return the Phase 4-selected issue origins, deduplicated by SID/time."""
    filtered = filter_north_atlantic(load_ibtracs(csv_path))
    origins = []
    for horizon, steps in SELECTED_HISTORIES:
        origins.append(build_forecast_samples(filtered, horizon, history_steps=steps).metadata)
    combined = pd.concat(origins, ignore_index=True)
    combined = combined.sort_values(["SID", "issue_time", "target_time"])
    return combined.drop_duplicates(["SID", "issue_time"], keep="first").reset_index(drop=True)


def manifest_from_origins(origins: pd.DataFrame) -> pd.DataFrame:
    """Make the fixed September--October CDS manifest from Phase 4 issue timestamps."""
    issue_times = pd.DatetimeIndex(origins["issue_time"]).tz_convert("UTC")
    dates = pd.DatetimeIndex(issue_times.normalize().unique()).sort_values()
    dates = dates[dates.month.isin(ERA5_MONTHS)]
    manifest = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    manifest["times"] = ",".join(TIMES)
    manifest["cds_dataset_resolution"] = "0.25 degree regular latitude/longitude"
    manifest["area"] = json.dumps(AREA)
    return manifest


def file_specs(manifest: pd.DataFrame) -> list[FileSpec]:
    dates = pd.to_datetime(manifest["date"], utc=True)
    groups = pd.DataFrame({"year": dates.dt.year, "month": dates.dt.month, "day": dates.dt.strftime("%d")})
    specs = []
    for (year, month), group in groups.groupby(["year", "month"], sort=True):
        for product in PRODUCTS:
            specs.append(FileSpec(product, int(year), int(month), tuple(group["day"])))
    return specs


def expected_times(spec: FileSpec) -> pd.DatetimeIndex:
    if spec.request_payload is not None:
        payload = _spec_payload(spec)
        return pd.DatetimeIndex(
            [f"{payload['year'][0]}-{payload['month'][0]}-{payload['day'][0]}T{hour}:00Z"
             for hour in payload["time"]],
            tz="UTC",
        )
    return pd.DatetimeIndex(
        [f"{spec.year:04d}-{spec.month:02d}-{day}T{hour}:00Z" for day in spec.days for hour in TIMES],
        tz="UTC",
    )


def _spec_payload(spec: FileSpec) -> dict[str, Any]:
    """Return the exact canonical CDS payload assigned to this durable file identity."""
    if spec.request_payload is not None:
        return _canonical_payload(spec.request_payload)
    product = PRODUCTS[spec.product]
    payload: dict[str, Any] = {
        "product_type": ["reanalysis"], "variable": product["variable"],
        "year": [str(spec.year)], "month": [f"{spec.month:02d}"], "day": list(spec.days),
        "time": TIMES, "area": AREA, "data_format": "netcdf", "download_format": "unarchived",
    }
    if "pressure_level" in product:
        payload["pressure_level"] = product["pressure_level"]
    return _canonical_payload(payload)


def _spec_dataset(spec: FileSpec) -> str:
    if spec.request_dataset is not None:
        return spec.request_dataset
    return PRODUCTS[spec.product]["dataset"]


def _spec_product(spec: FileSpec) -> dict[str, Any]:
    return GENESIS_PRODUCTS[spec.product] if spec.request_payload is not None else PRODUCTS[spec.product]


def _units(value: str | None) -> str:
    return (value or "").replace("**", "").replace(" ", "")


def _coordinate(dataset: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in dataset.variables:
            return dataset.variables[name]
    raise ValueError(f"missing coordinate; expected one of {names}")


def _timestamps(variable: Any) -> pd.DatetimeIndex:
    from netCDF4 import num2date

    values = num2date(
        variable[:], variable.units, calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False, only_use_python_datetimes=True,
    )
    return pd.DatetimeIndex(values, tz="UTC")


def validate_file(path: Path, spec: FileSpec) -> dict[str, Any]:
    """Fail closed unless a raw CDS file exactly matches its durable request spec."""
    from netCDF4 import Dataset

    product = _spec_product(spec)
    with Dataset(path) as dataset:
        time = _coordinate(dataset, ("valid_time", "time"))
        latitude = _coordinate(dataset, ("latitude", "lat"))
        longitude = _coordinate(dataset, ("longitude", "lon"))
        actual_times = _timestamps(time)
        expected = expected_times(spec)
        if actual_times.has_duplicates:
            raise ValueError(f"{path}: duplicate timestamps")
        if set(actual_times) != set(expected):
            missing = expected.difference(actual_times).astype(str).tolist()
            extra = actual_times.difference(expected).astype(str).tolist()
            raise ValueError(f"{path}: missing timestamps={missing}, unexpected timestamps={extra}")

        latitudes = np.asarray(latitude[:], dtype=float)
        longitudes = np.asarray(longitude[:], dtype=float)
        if len(np.unique(latitudes)) != len(latitudes) or len(np.unique(longitudes)) != len(longitudes):
            raise ValueError(f"{path}: duplicate latitude or longitude coordinates")
        area = _spec_payload(spec)["area"]
        if not (np.isclose(latitudes.min(), area[2]) and np.isclose(latitudes.max(), area[0])):
            raise ValueError(f"{path}: latitude coverage is {latitudes.min()}..{latitudes.max()}")
        if not np.allclose(np.abs(np.diff(latitudes)), 0.25):
            raise ValueError(f"{path}: latitude is not a 0.25 degree regular grid")
        if spec.request_payload is None:
            if not (np.isclose(longitudes.min(), area[1]) and np.isclose(longitudes.max(), area[3])):
                raise ValueError(f"{path}: longitude coverage is {longitudes.min()}..{longitudes.max()}")
            if not np.allclose(np.abs(np.diff(longitudes)), 0.25):
                raise ValueError(f"{path}: longitude is not a 0.25 degree regular grid")
        else:
            # CDS may emit either signed or 0..360 longitudes after an API dateline subset.
            # Relative circular offsets preserve the requested west-to-east interval in both forms.
            offsets = np.sort((longitudes - area[1]) % 360)
            width = (area[3] - area[1]) % 360
            expected_offsets = np.arange(round(width / 0.25) + 1) * 0.25
            if len(offsets) != len(expected_offsets) or not np.allclose(offsets, expected_offsets):
                raise ValueError(f"{path}: longitude coverage does not match CDS area {area}")

        variables = set(dataset.variables)
        if not product["data_variables"].issubset(variables):
            raise ValueError(f"{path}: data variables are {sorted(variables)}")
        for name, expected_unit in product["units"].items():
            if _units(getattr(dataset.variables[name], "units", None)) != _units(expected_unit):
                raise ValueError(f"{path}: {name} units are {getattr(dataset.variables[name], 'units', None)!r}")

        levels: list[float] = []
        if "pressure_level" in product:
            level = _coordinate(dataset, ("pressure_level", "level"))
            levels = np.asarray(level[:], dtype=float).tolist()
            wanted = sorted(float(value) for value in product["pressure_level"])
            if sorted(levels) != wanted:
                raise ValueError(f"{path}: pressure levels are {levels}, expected {wanted}")

    return {
        "variables": sorted(product["data_variables"]),
        "levels": levels,
        "timestamps": len(actual_times),
        "latitude": [float(latitudes.min()), float(latitudes.max())],
        "longitude": [float(longitudes.min()), float(longitudes.max())],
        "bytes": path.stat().st_size,
    }


def _journal_path(state_dir: Path) -> Path:
    return state_dir / "request_jobs.json"


def _load_journal(state_dir: Path) -> dict[str, dict[str, Any]]:
    path = _journal_path(state_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def _save_journal(state_dir: Path, journal: dict[str, dict[str, Any]]) -> None:
    """Durably replace the journal so a crash cannot expose a partial JSON document."""
    path = _journal_path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=state_dir)
    try:
        if path.exists():
            os.fchmod(descriptor, stat.S_IMODE(path.stat().st_mode))
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(journal, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(state_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        except OSError:  # Some filesystems do not support syncing directories.
            pass
        finally:
            os.close(directory)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


@contextmanager
def acquisition_lock(state_dir: Path):
    """Reject a second runner before it can inspect or submit a CDS request."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "acquire.lock"
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another ERA5 acquisition runner holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def journal_jobs(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize legacy entries but reject any journal state without an ID."""
    if entry is None:
        return []
    if "jobs" in entry:
        jobs = entry["jobs"]
        if not isinstance(jobs, list) or not jobs or any(not job.get("request_id") for job in jobs):
            raise RuntimeError(f"ambiguous CDS journal entry: {entry}")
        return jobs
    if entry.get("request_id"):
        return [{key: value for key, value in entry.items() if key != "jobs"}]
    raise RuntimeError(f"ambiguous CDS journal entry: {entry}")


def journal_entry(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every known request ID; multiple jobs are recovery-only, never resubmitted."""
    if not jobs or any(not job.get("request_id") for job in jobs):
        raise ValueError("journal entries require at least one CDS request ID")
    return {"jobs": jobs}


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON-compatible CDS inputs so map ordering cannot change identity."""
    return json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    canonical = _canonical_payload(payload)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def genesis_file_specs(path: Path) -> list[FileSpec]:
    """Load only the approved, canonical TCC CDS requests without altering them."""
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or not isinstance(document.get("requests"), list):
        raise ValueError(f"{path}: unsupported genesis ERA5 request manifest")
    specs = []
    seen = set()
    for request in document["requests"]:
        required = {"request_key", "tcc_track_id", "utc_day", "product", "dataset", "payload", "payload_sha256"}
        if required - set(request):
            raise ValueError(f"{path}: request lacks {sorted(required - set(request))}")
        product = request["product"]
        if product not in GENESIS_PRODUCTS:
            raise ValueError(f"{path}: unsupported genesis product {product!r}")
        payload = _canonical_payload(request["payload"])
        expected = GENESIS_PRODUCTS[product]
        payload_keys = {"product_type", "variable", "year", "month", "day", "time", "area", "data_format", "download_format"}
        if product == "pressure":
            payload_keys.add("pressure_level")
        if set(payload) != payload_keys:
            raise ValueError(f"{path}: unexpected CDS payload fields for {request['request_key']}")
        if request["dataset"] != expected["dataset"] or payload.get("variable") != expected["variable"]:
            raise ValueError(f"{path}: unexpected dataset or variable for {request['request_key']}")
        if product == "pressure" and payload.get("pressure_level") != expected["pressure_level"]:
            raise ValueError(f"{path}: unexpected pressure levels for {request['request_key']}")
        if payload.get("product_type") != ["reanalysis"] or payload.get("data_format") != "netcdf" or payload.get("download_format") != "unarchived":
            raise ValueError(f"{path}: unexpected CDS format for {request['request_key']}")
        if (not isinstance(payload.get("time"), list) or not payload["time"]
                or payload["time"] != sorted(set(payload["time"]))
                or any(hour not in {f"{value:02d}:00" for value in range(0, 24, 3)} for hour in payload["time"])):
            raise ValueError(f"{path}: unexpected frozen TCC times for {request['request_key']}")
        area = payload.get("area")
        if (not isinstance(area, list) or len(area) != 4 or not all(isinstance(value, (int, float)) for value in area)
                or not (-90 <= area[2] <= area[0] <= 90 and -180 <= area[1] <= 180 and -180 <= area[3] <= 180)
                or (area[3] - area[1]) % 360 >= 30 or area[0] - area[2] >= 30):
            raise ValueError(f"{path}: invalid regional CDS area for {request['request_key']}")
        if request["payload_sha256"] != _payload_fingerprint(payload):
            raise ValueError(f"{path}: payload fingerprint mismatch for {request['request_key']}")
        try:
            date = pd.Timestamp(request["utc_day"], tz="UTC")
        except ValueError as error:
            raise ValueError(f"{path}: invalid UTC day for {request['request_key']}") from error
        if ([date.strftime("%Y")], [date.strftime("%m")], [date.strftime("%d")]) != (
            payload.get("year"), payload.get("month"), payload.get("day")
        ):
            raise ValueError(f"{path}: date does not match payload for {request['request_key']}")
        key = request["request_key"]
        expected_key = f"genesis_{product}_{request['tcc_track_id']}_{date:%Y%m%d}"
        if key != expected_key or key in seen:
            raise ValueError(f"{path}: non-deterministic or duplicate request key {key!r}")
        seen.add(key)
        specs.append(FileSpec(product, date.year, date.month, (date.strftime("%d"),), key, request["dataset"], payload))
    if not specs:
        raise ValueError(f"{path}: no genesis ERA5 requests")
    return specs


def genesis_smoke_specs(specs: list[FileSpec]) -> list[FileSpec]:
    """Select one approved pressure request and its same track/day SST partner."""
    pressure = next((spec for spec in specs if spec.product == "pressure"), None)
    if pressure is None or pressure.request_key is None:
        raise ValueError("genesis manifest has no pressure request for the smoke test")
    sst_key = pressure.request_key.replace("genesis_pressure_", "genesis_sst_", 1)
    sst = next((spec for spec in specs if spec.request_key == sst_key), None)
    if sst is None:
        raise ValueError(f"genesis manifest lacks SST partner for {pressure.request_key}")
    return [pressure, sst]


def _ambiguous_match_fields(entry: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    payload = entry.get("submission_payload")
    fingerprint = entry.get("submission_fingerprint")
    dataset = entry.get("submission_dataset")
    if not isinstance(payload, dict) or not isinstance(fingerprint, str) or not isinstance(dataset, str):
        raise RuntimeError("ambiguous CDS submission lacks a recoverable canonical request")
    if _payload_fingerprint(payload) != fingerprint:
        raise RuntimeError("ambiguous CDS submission has a mismatched request fingerprint")
    return dataset, payload, fingerprint


def _matching_ambiguous_jobs(client: Any, entry: dict[str, Any]) -> list[Any]:
    """Return only authoritative CDS jobs matching the persisted submission attempt."""
    dataset, payload, fingerprint = _ambiguous_match_fields(entry)

    matches = []
    seen = set()
    page = client.client.get_jobs(limit=100, sortby="-created")
    while page is not None:
        for request_id in page.request_ids:
            if request_id in seen:
                continue
            seen.add(request_id)
            remote = client.client.get_remote(request_id)
            if remote.collection_id != dataset:
                continue
            remote_payload = _canonical_payload(remote.request)
            if remote_payload == payload and _payload_fingerprint(remote_payload) == fingerprint:
                matches.append(remote)
        page = page.next
    return matches


def _ambiguous_job_snapshot(client: Any) -> list[dict[str, Any]]:
    """Read CDS history once so Genesis reconciliation does not repeat account-wide scans."""
    snapshot = []
    seen = set()
    page = client.client.get_jobs(limit=100, sortby="-created")
    pages = 0
    while page is not None:
        pages += 1
        if pages > AMBIGUOUS_HISTORY_MAX_PAGES:
            raise RuntimeError("CDS history pagination limit exceeded")
        for request_id in page.request_ids:
            if request_id in seen:
                continue
            seen.add(request_id)
            remote = client.client.get_remote(request_id)
            metadata = remote.json
            payload = _canonical_payload(dict(metadata["metadata"]["request"]["ids"]))
            snapshot.append({
                "remote": remote,
                "dataset": str(metadata["processID"]),
                "payload": payload,
                "fingerprint": _payload_fingerprint(payload),
            })
        page = page.next
    return snapshot


def _matching_snapshot_jobs(snapshot: list[dict[str, Any]], entry: dict[str, Any]) -> list[Any]:
    """Apply the same canonical dataset/payload/fingerprint match to one history snapshot."""
    dataset, payload, fingerprint = _ambiguous_match_fields(entry)
    return [record["remote"] for record in snapshot
            if record["dataset"] == dataset and record["payload"] == payload
            and record["fingerprint"] == fingerprint]


def _proven_pre_cds_dns_failure(entry: dict[str, Any]) -> bool:
    error = entry.get("submission_error")
    return (isinstance(error, str) and "NameResolutionError" in error
            and "Failed to resolve" in error and "cds.climate.copernicus.eu" in error)


def _journaled_job_owner(journal: dict[str, dict[str, Any]], request_id: str) -> str | None:
    for key, entry in journal.items():
        jobs = entry.get("jobs")
        if isinstance(jobs, list) and any(job.get("request_id") == request_id for job in jobs):
            return key
        if entry.get("request_id") == request_id:
            return key
    return None


def _journal_payload_matches_spec(entry: dict[str, Any], spec: FileSpec) -> bool:
    payload = _canonical_payload(spec.request_payload)
    return (entry.get("submission_dataset") == spec.request_dataset
            and entry.get("submission_payload") == payload
            and entry.get("submission_fingerprint") == _payload_fingerprint(payload))


def _reconcile_genesis_ambiguities(
    specs: list[FileSpec], journal: dict[str, dict[str, Any]], state_dir: Path
) -> set[str]:
    """Resolve every uncertain Genesis POST before the coordinator can make another POST."""
    unresolved = set()
    candidates = []
    for spec in specs:
        if spec.request_payload is None:
            continue
        key = spec.filename
        entry = journal.get(key)
        if not entry or "submission_error" not in entry or "jobs" in entry:
            continue
        if _proven_pre_cds_dns_failure(entry):
            continue
        if not _journal_payload_matches_spec(entry, spec):
            unresolved.add(key)
            continue
        candidates.append((key, entry))

    if not candidates:
        return unresolved
    try:
        snapshot = _ambiguous_job_snapshot(_submission_client(None))
    except Exception:
        # A history lookup failure is itself ambiguous; leave original entries untouched
        # and allow independent Genesis work to continue without a POST.
        unresolved.update(key for key, _entry in candidates)
        return unresolved

    for key, entry in candidates:
        try:
            matches = _matching_snapshot_jobs(snapshot, entry)
        except Exception:
            unresolved.add(key)
            continue
        if len(matches) != 1:
            unresolved.add(key)
            continue
        remote = matches[0]
        request_id = remote.request_id
        if not request_id or _journaled_job_owner(journal, request_id) is not None:
            unresolved.add(key)
            continue
        job = {"request_id": request_id, "status": "recovered_submission"}
        if hasattr(remote, "url"):
            job["monitor_url"] = remote.url
        entry["jobs"] = [job]
        journal[key] = entry
        _save_journal(state_dir, journal)
    return unresolved


def _history_before_dns_retry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    history = entry.get("submission_history", [])
    if not isinstance(history, list):
        raise RuntimeError("invalid Genesis submission history")
    previous = {key: value for key, value in entry.items() if key not in {"jobs", "submission_history"}}
    return [*history, previous]


def _request_payload(spec: FileSpec) -> dict[str, Any]:
    return _spec_payload(spec)


def _cds_client(*, genesis_submission: bool = False) -> Any:
    """Use one bounded transport attempt; recovery happens on the next journaled invocation."""
    import cdsapi

    options = CDS_CLIENT_OPTIONS
    if genesis_submission:
        import requests

        options = {**CDS_CLIENT_OPTIONS, "timeout": (10, 90), "session": requests.Session()}
    return cdsapi.Client(quiet=False, progress=False, wait_until_complete=False, **options)


def _download_with_deadline(url: str, target: Path) -> None:
    """Resume the CDS asset with curl; curl owns the transfer and retry deadlines."""
    try:
        subprocess.run([
            "curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
            "--retry-all-errors", "--retry-delay", "5", "--retry-max-time", "900",
            "--max-time", "900", "--output", str(target), url,
        ], check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"curl failed with status {error.returncode}") from error


def _wait_and_download(remote: Any, target: Path) -> int:
    """Use the datastore job monitor returned by CDS, never legacy /api/tasks URLs."""
    delay = 2.0
    while True:
        status = remote.status
        if status == "successful":
            results = remote.get_results()
            expected_bytes = results.content_length
            if not target.exists() or target.stat().st_size != expected_bytes:
                _download_with_deadline(results.location, target)
            if target.stat().st_size != expected_bytes:
                raise RuntimeError(f"downloaded {target.stat().st_size} bytes, expected {expected_bytes}")
            return expected_bytes
        if status in {"failed", "rejected", "dismissed", "deleted"}:
            try:
                receipt = remote.get_receipt()
            except Exception as error:  # Keep the server failure, even if receipt lookup also fails.
                receipt = {"receipt_error": str(error)}
            raise RuntimeError(f"CDS job {remote.request_id} ended as {status}: {receipt}")
        time.sleep(delay)
        delay = min(delay * 1.5, 60.0)


def _invalid_path(target: Path) -> Path:
    suffix = 0
    while True:
        path = target.with_suffix(target.suffix + f".invalid{suffix or ''}")
        if not path.exists():
            return path
        suffix += 1


def _recover_job(client: Any, job: dict[str, Any], target: Path, spec: FileSpec) -> None:
    """Retry only a recorded CDS asset, preserving its partial file for HTTP Range resume."""
    remote = client.client.get_remote(job["request_id"])
    temporary = target.with_suffix(target.suffix + f".{remote.request_id}.part")
    errors = []
    for _ in range(DOWNLOAD_ATTEMPTS):
        try:
            _wait_and_download(remote, temporary)
            validate_file(temporary, spec)
        except Exception as error:
            errors.append(str(error))
            continue
        temporary.replace(target)
        job.update(status="validated", bytes=target.stat().st_size)
        return
    raise RuntimeError(f"CDS download failed after {DOWNLOAD_ATTEMPTS} attempts: {errors}")


def _recover_recorded_jobs(raw_dir: Path, spec: FileSpec, jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Recover one file with a private CDS client; the coordinator persists its result."""
    chosen = None
    try:
        client = _cds_client()
        statuses = {job["request_id"]: client.client.get_remote(job["request_id"]).status for job in jobs}
        successful = [job for job in jobs if statuses[job["request_id"]] == "successful"]
        active = [job for job in jobs if statuses[job["request_id"]] in {"accepted", "running"}]
        if successful:
            chosen = successful[0]
        elif active:
            chosen = active[0]
        else:
            raise RuntimeError(f"all recorded CDS jobs need diagnosis for {spec.filename}: {statuses}")
        _recover_job(client, chosen, raw_dir / spec.filename, spec)
    except Exception as error:
        # A worker never writes the shared journal.  It returns the exact existing IDs to
        # the lock-owning coordinator, so a transient failure cannot become a resubmission.
        if chosen is not None:
            chosen["download_error"] = str(error)
        else:
            for job in jobs:
                job["last_error"] = str(error)
        return jobs, str(error)
    return jobs, None


def _submission_client(client: Any | None, *, genesis_submission: bool = False) -> Any:
    if genesis_submission:
        # One fresh pool per Genesis POST avoids reusing a potentially stale connection.
        return _cds_client(genesis_submission=True)
    return client if client is not None else _cds_client()


def acquire(
    raw_dir: Path,
    specs: list[FileSpec],
    concurrency: int = ACQUISITION_CONCURRENCY,
    new_request_limit: int | None = None,
) -> dict[str, Any]:
    """Download into raw_dir while one canonical project journal owns CDS job identity."""
    if concurrency < 1:
        raise ValueError("ERA5 acquisition concurrency must be positive")
    if new_request_limit is not None and new_request_limit < 1:
        raise ValueError("new request limit must be positive")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with acquisition_lock(ERA5_STATE_DIR):
        journal = _load_journal(ERA5_STATE_DIR)
        genesis_run = bool(specs) and all(spec.request_payload is not None for spec in specs)
        genesis_ambiguous = (
            _reconcile_genesis_ambiguities(specs, journal, ERA5_STATE_DIR) if genesis_run else set()
        )
        if new_request_limit is not None:
            if any(spec.request_payload is None for spec in specs):
                raise ValueError("new request limits are supported only for Genesis manifest requests")
            specs = [spec for spec in specs
                     if spec.filename not in journal and not (raw_dir / spec.filename).exists()][:new_request_limit]
        submission_client = None
        counts = Counter(requested=len(specs))
        unresolved_requests = []
        unresolved_keys = set()

        def mark_unresolved(key: str) -> None:
            if key not in unresolved_keys:
                unresolved_keys.add(key)
                unresolved_requests.append(key)
                counts["unresolved"] += 1

        for spec in specs:
            if spec.filename in genesis_ambiguous:
                mark_unresolved(spec.filename)
        pending = iter(specs)
        while True:
            batch: list[tuple[FileSpec, dict[str, Any], list[dict[str, Any]]]] = []
            while len(batch) < concurrency:
                try:
                    spec = next(pending)
                except StopIteration:
                    break
                target, key = raw_dir / spec.filename, spec.filename
                entry = journal.get(key)
                dns_retry_history = None
                if spec.request_payload is not None and entry and "submission_error" in entry and "jobs" not in entry:
                    if not _proven_pre_cds_dns_failure(entry):
                        mark_unresolved(key)
                        continue
                    if not _journal_payload_matches_spec(entry, spec):
                        mark_unresolved(key)
                        continue
                    # A recorded DNS failure cannot have reached CDS; retain it before a fresh attempt.
                    dns_retry_history = _history_before_dns_retry(entry)

                if target.exists():
                    try:
                        validate_file(target, spec)
                        counts["skipped_valid"] += 1
                        continue
                    except Exception as error:
                        # Preserve evidence; only a recorded CDS job may replace a corrupt target.
                        target.replace(_invalid_path(target))
                        journal.setdefault(key, {})["invalid_local_file"] = str(error)
                        _save_journal(ERA5_STATE_DIR, journal)

                if spec.request_payload is None and entry and "submission_error" in entry and "jobs" not in entry:
                    submission_client = _submission_client(submission_client)
                    matches = _matching_ambiguous_jobs(submission_client, entry)
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"ambiguous CDS submission for {key}: found {len(matches)} matching CDS jobs; will not resubmit"
                        )
                    remote = matches[0]
                    job = {"request_id": remote.request_id, "status": "recovered_submission"}
                    if hasattr(remote, "url"):
                        job["monitor_url"] = remote.url
                    entry["jobs"] = [job]
                    journal[key] = entry
                    _save_journal(ERA5_STATE_DIR, journal)  # Persist the recovered ID before recovery.

                jobs = [] if dns_retry_history is not None else journal_jobs(entry)
                if not jobs:
                    # No entry is the only state in which submission is allowed.  Persist the
                    # exact attempt first because a transport error may conceal CDS acceptance.
                    payload = _canonical_payload(_request_payload(spec))
                    dataset = _spec_dataset(spec)
                    entry = {
                        "submission_payload": payload,
                        "submission_fingerprint": _payload_fingerprint(payload),
                        "submission_dataset": dataset,
                    }
                    if dns_retry_history is not None:
                        entry["submission_history"] = dns_retry_history
                    journal[key] = entry
                    _save_journal(ERA5_STATE_DIR, journal)
                    submission_client = _submission_client(
                        submission_client, genesis_submission=spec.request_payload is not None
                    )
                    try:
                        remote = submission_client.retrieve(dataset, payload)
                        request_id = remote.request_id
                        if not request_id:
                            raise RuntimeError("CDS submission returned no job ID")
                    except Exception as error:
                        entry["submission_error"] = str(error)
                        journal[key] = entry
                        _save_journal(ERA5_STATE_DIR, journal)
                        if spec.request_payload is None:
                            raise RuntimeError(f"ambiguous CDS submission for {key}; inspect before retrying") from error
                        unresolved_requests.append(key)
                        counts["unresolved"] += 1
                        continue
                    if genesis_run and _journaled_job_owner(journal, request_id) is not None:
                        entry["submission_error"] = "CDS returned a job ID already recorded for another request"
                        entry["duplicate_request_id"] = request_id
                        journal[key] = entry
                        _save_journal(ERA5_STATE_DIR, journal)
                        mark_unresolved(key)
                        continue
                    jobs = [{"request_id": request_id, "status": "submitted"}]
                    journal[key] = {**entry, "jobs": jobs}
                    _save_journal(ERA5_STATE_DIR, journal)  # Persist before any worker polls/downloads.
                    entry = journal[key]

                # Copy nested jobs: workers may update their chosen job, but only this coordinator saves it.
                batch.append((spec, entry, [dict(job) for job in jobs]))

            if not batch:
                break
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_recover_recorded_jobs, raw_dir, spec, jobs): (spec, entry)
                           for spec, entry, jobs in batch}
                for future in as_completed(futures):
                    spec, entry = futures[future]
                    jobs, error = future.result()
                    journal[spec.filename] = {**entry, "jobs": jobs}
                    _save_journal(ERA5_STATE_DIR, journal)
                    if error:
                        counts["failed"] += 1
                    else:
                        counts["downloaded"] += 1
        summary = {**counts, "unresolved_requests": unresolved_requests}
        if new_request_limit is not None:
            summary["scope_requests"] = [spec.filename for spec in specs]
        return summary


def _value(variable: Any, indexes: dict[str, int]) -> float | None:
    selection = []
    for dimension in variable.dimensions:
        if dimension in indexes:
            selection.append(indexes[dimension])
        elif variable.shape[len(selection)] == 1:
            selection.append(0)
        else:
            raise ValueError(f"unsupported non-singleton dimension {dimension!r} in {variable.name}")
    value = variable[tuple(selection)]
    if np.ma.is_masked(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _indexes(dataset: Any, issue_time: pd.Timestamp, lat: float, lon: float) -> dict[str, int]:
    time = _coordinate(dataset, ("valid_time", "time"))
    latitude = _coordinate(dataset, ("latitude", "lat"))
    longitude = _coordinate(dataset, ("longitude", "lon"))
    times = _timestamps(time)
    matches = np.flatnonzero(times == issue_time)
    if len(matches) != 1:
        raise ValueError(f"ERA5 timestamp lookup returned {len(matches)} matches for {issue_time.isoformat()}")
    return {
        time.dimensions[0]: int(matches[0]),
        latitude.dimensions[0]: int(np.abs(np.asarray(latitude[:]) - lat).argmin()),
        longitude.dimensions[0]: int(np.abs(np.asarray(longitude[:]) - lon).argmin()),
    }


def extract_features(raw_dir: Path, origins: pd.DataFrame, output: Path) -> dict[str, Any]:
    """Extract exact-time, nearest-point scalar ERA5 predictors for selected origins."""
    from netCDF4 import Dataset

    rows = []
    missing = Counter()
    for month, group in origins.groupby(origins["issue_time"].dt.strftime("%Y%m"), sort=True):
        specs = {product: FileSpec(product, int(month[:4]), int(month[4:]), ()) for product in PRODUCTS}
        paths = {product: raw_dir / spec.filename for product, spec in specs.items()}
        for product, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing validated ERA5 file: {path}")
        with Dataset(paths["pressure_wind"]) as wind, Dataset(paths["humidity_700"]) as humidity, Dataset(paths["sst"]) as sst:
            wind_level = _coordinate(wind, ("pressure_level", "level"))
            levels = {int(value): index for index, value in enumerate(np.asarray(wind_level[:], dtype=int))}
            humidity_level = _coordinate(humidity, ("pressure_level", "level"))
            humidity_levels = {int(value): index for index, value in enumerate(np.asarray(humidity_level[:], dtype=int))}
            for origin in group.itertuples(index=False):
                issue_time = pd.Timestamp(origin.issue_time)
                wind_index = _indexes(wind, issue_time, origin.issue_lat, origin.issue_lon)
                humidity_index = _indexes(humidity, issue_time, origin.issue_lat, origin.issue_lon)
                sst_index = _indexes(sst, issue_time, origin.issue_lat, origin.issue_lon)
                wind_index[wind_level.dimensions[0]] = levels[850]
                u850 = _value(wind.variables["u"], wind_index)
                v850 = _value(wind.variables["v"], wind_index)
                wind_index[wind_level.dimensions[0]] = levels[700]
                u700 = _value(wind.variables["u"], wind_index)
                v700 = _value(wind.variables["v"], wind_index)
                wind_index[wind_level.dimensions[0]] = levels[500]
                u500 = _value(wind.variables["u"], wind_index)
                v500 = _value(wind.variables["v"], wind_index)
                wind_index[wind_level.dimensions[0]] = levels[200]
                u200 = _value(wind.variables["u"], wind_index)
                v200 = _value(wind.variables["v"], wind_index)
                humidity_index[humidity_level.dimensions[0]] = humidity_levels[700]
                relative_humidity = _value(humidity.variables["r"], humidity_index)
                sea_surface_temperature = _value(sst.variables["sst"], sst_index)

                values = {"u850": u850, "v850": v850, "u700": u700, "v700": v700, "u500": u500, "v500": v500,
                          "u200": u200, "v200": v200, "relative_humidity_700": relative_humidity,
                          "sea_surface_temperature": sea_surface_temperature}
                missing_fields = [name for name, value in values.items() if value is None]
                for name in missing_fields:
                    missing[name] += 1
                complete = not missing_fields
                latitude = _coordinate(wind, ("latitude", "lat"))
                longitude = _coordinate(wind, ("longitude", "lon"))
                row = {
                    "SID": origin.SID, "issue_time": issue_time.isoformat(), "issue_lat": origin.issue_lat, "issue_lon": origin.issue_lon,
                    "era5_time": issue_time.isoformat(), "era5_grid_lat": float(latitude[wind_index[latitude.dimensions[0]]]),
                    "era5_grid_lon": float(longitude[wind_index[longitude.dimensions[0]]]), "era5_available": complete,
                    "missing_era5_fields": ",".join(missing_fields),
                    "steering_u": (u850 + u700 + u500) / 3 if all(value is not None for value in (u850, u700, u500)) else np.nan,
                    "steering_v": (v850 + v700 + v500) / 3 if all(value is not None for value in (v850, v700, v500)) else np.nan,
                    "vertical_shear": float(np.hypot(u200 - u850, v200 - v850)) if all(value is not None for value in (u200, u850, v200, v850)) else np.nan,
                    "relative_humidity_700": relative_humidity if relative_humidity is not None else np.nan,
                    "sea_surface_temperature": sea_surface_temperature if sea_surface_temperature is not None else np.nan,
                }
                rows.append(row)

    features = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    feature_columns = ("steering_u", "steering_v", "vertical_shear", "relative_humidity_700", "sea_surface_temperature")
    return {
        "origins": len(features), "complete_origins": int(features["era5_available"].sum()),
        "missing_origins": int((~features["era5_available"]).sum()), "storms": int(features["SID"].nunique()),
        "date_range": [features["issue_time"].min(), features["issue_time"].max()],
        "missingness_by_feature": {name: int(features[name].isna().sum()) for name in feature_columns},
        "missingness_by_raw_field": dict(missing),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="?", help="official ibtracs.NA.list.v04r01.csv")
    parser.add_argument("--genesis-manifest", type=Path,
                        help="approved metadata-only TCC genesis ERA5 request manifest")
    parser.add_argument("--smoke-test", action="store_true",
                        help="with --genesis-manifest, acquire only one pressure request and its SST partner")
    parser.add_argument("--limit-new", type=int,
                        help="with --genesis-manifest, process at most this many requests absent from the journal and raw directory")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/era5/v1"))
    parser.add_argument("--output", type=Path, default=Path("results/phase5_era5_features.csv"))
    parser.add_argument("--concurrency", type=int, default=ACQUISITION_CONCURRENCY,
                        help=f"maximum concurrent CDS recoveries (default: {ACQUISITION_CONCURRENCY})")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    if args.genesis_manifest:
        if args.csv:
            parser.error("csv and --genesis-manifest are separate acquisition modes")
        if args.limit_new is not None and args.smoke_test:
            parser.error("--limit-new and --smoke-test cannot be combined")
        specs = genesis_file_specs(args.genesis_manifest)
        if args.smoke_test:
            specs = genesis_smoke_specs(specs)
        acquisition = acquire(args.raw_dir, specs, args.concurrency, new_request_limit=args.limit_new)
        if acquisition.get("failed"):
            print(json.dumps({"acquisition": acquisition}, indent=2))
            raise SystemExit(1)
        unresolved = set(acquisition["unresolved_requests"])
        scoped = acquisition.get("scope_requests")
        validations = [validate_file(args.raw_dir / spec.filename, spec)
                       for spec in specs if spec.filename not in unresolved and
                       (scoped is None or spec.filename in scoped)]
        print(json.dumps({"acquisition": acquisition, "validation": {"files": len(validations), "passed": True}}, indent=2))
        return
    if not args.csv:
        parser.error("csv is required unless --genesis-manifest is supplied")
    if args.smoke_test:
        parser.error("--smoke-test requires --genesis-manifest")
    if args.limit_new is not None:
        parser.error("--limit-new requires --genesis-manifest")
    manifest_path = args.raw_dir / ERA5_MANIFEST_NAME
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = manifest_from_origins(selected_origins(args.csv))
        manifest.to_csv(manifest_path, index=False)
    if len(manifest) != 1339 or manifest.iloc[0]["date"] != "1980-09-02" or manifest.iloc[-1]["date"] != "2025-10-31":
        raise ValueError("ERA5 manifest does not match the approved September--October plan")
    specs = file_specs(manifest)
    if len(specs) != 264:
        raise ValueError("ERA5 manifest does not produce the approved 264 file specs")
    acquisition = acquire(args.raw_dir, specs, args.concurrency)
    if acquisition.get("failed"):
        print(json.dumps({"acquisition": acquisition}, indent=2))
        raise SystemExit(1)

    # Every file is rechecked before extraction, including files skipped during resume.
    validations = [validate_file(args.raw_dir / spec.filename, spec) for spec in specs]
    origins = selected_origins(args.csv)
    origins = origins[pd.to_datetime(origins["issue_time"], utc=True).dt.month.isin(ERA5_MONTHS)].reset_index(drop=True)
    summary = {"acquisition": acquisition, "validation": {"files": len(validations), "passed": True},
               "features": extract_features(args.raw_dir, origins, args.output)}
    summary_path = args.output.with_name("phase5_era5_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
