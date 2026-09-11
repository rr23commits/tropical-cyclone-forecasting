"""Load, filter, split, and window the audited IBTrACS North Atlantic data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = (
    "SID", "SEASON", "NAME", "BASIN", "ISO_TIME", "TRACK_TYPE", "USA_LAT",
    "USA_LON", "USA_WIND", "USA_PRES", "USA_AGENCY", "USA_STATUS", "IFLAG",
)
CORE_STATE_COLUMNS = ("USA_LAT", "USA_LON", "USA_WIND")
TROPICAL_STATUSES = frozenset({"TD", "TS", "HU", "TY", "ST", "TC", "SD", "SS"})
FEATURE_NAMES = (
    "latitude", "longitude", "wind", "north_step_km", "east_step_km",
    "speed_kmh", "direction_sin", "direction_cos", "wind_change",
)


@dataclass(frozen=True)
class ForecastSamples:
    """Feature windows and direct targets for one forecast horizon."""

    features: np.ndarray
    targets: np.ndarray  # north displacement (km), east displacement (km), wind change
    metadata: pd.DataFrame
    history_times: np.ndarray
    horizon_hours: int


def load_ibtracs(path: str | Path) -> pd.DataFrame:
    """Load the official basin CSV, whose second row contains units."""
    data = pd.read_csv(
        path,
        skiprows=[1],
        usecols=lambda column: column in CORE_COLUMNS,
        low_memory=False,
        # `NA` is a real basin code, so it must not be interpreted as missing.
        keep_default_na=False,
        na_values=[" "],
    )
    missing = set(CORE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"IBTrACS CSV is missing columns: {sorted(missing)}")
    data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"], errors="coerce", utc=True)
    data["SEASON"] = pd.to_numeric(data["SEASON"], errors="coerce")
    for column in ("USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def filter_north_atlantic(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed Phase 1 North Atlantic, HURDAT, and six-hour rules."""
    missing = set(CORE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"data is missing columns: {sorted(missing)}")

    filtered = data.copy()
    timestamp = filtered["ISO_TIME"]
    is_six_hour = (
        timestamp.notna()
        & timestamp.dt.minute.eq(0)
        & timestamp.dt.second.eq(0)
        & timestamp.dt.hour.mod(6).eq(0)
    )
    keep = (
        filtered["BASIN"].eq("NA")
        & filtered["TRACK_TYPE"].eq("main")
        & filtered["SEASON"].between(1980, 2025)
        & filtered["USA_AGENCY"].eq("hurdat_atl")
        & filtered["USA_STATUS"].isin(TROPICAL_STATUSES)
        & is_six_hour
        & filtered.loc[:, CORE_STATE_COLUMNS].notna().all(axis=1)
    )
    filtered = filtered.loc[keep].copy()

    # A duplicate state has no authoritative row selection rule, so reject both rows.
    duplicates = filtered.duplicated(["SID", "ISO_TIME"], keep=False)
    filtered = filtered.loc[~duplicates].sort_values(["SID", "ISO_TIME"])
    return filtered.reset_index(drop=True)


def _displacement_km(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Return local north/east displacement while using the shortest longitude change."""
    north = (lat2 - lat1) * 111.32
    longitude_change = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    mean_latitude = np.deg2rad((lat1 + lat2) / 2.0)
    east = longitude_change * 111.32 * np.cos(mean_latitude)
    return north, east


def _window_features(window: pd.DataFrame) -> np.ndarray:
    """Create features using only the ordered observations in one history window."""
    values: list[list[float]] = []
    previous = None
    for row in window.itertuples(index=False):
        lat, lon, wind = float(row.USA_LAT), float(row.USA_LON), float(row.USA_WIND)
        if previous is None:
            north = east = speed = direction_sin = direction_cos = wind_change = 0.0
        else:
            north, east = _displacement_km(previous[0], previous[1], lat, lon)
            distance = float(np.hypot(north, east))
            speed = distance / 6.0
            direction_sin = east / distance if distance else 0.0
            direction_cos = north / distance if distance else 0.0
            wind_change = wind - previous[2]
        values.append([lat, lon, wind, north, east, speed, direction_sin, direction_cos, wind_change])
        previous = (lat, lon, wind)
    return np.asarray(values, dtype=float)


def build_forecast_samples(
    data: pd.DataFrame, horizon_hours: int, history_steps: int = 5
) -> ForecastSamples:
    """Build direct, contiguous six-hour history windows and future-state targets."""
    if horizon_hours not in {6, 12, 24, 48}:
        raise ValueError("horizon_hours must be one of 6, 12, 24, or 48")
    if history_steps < 2:
        raise ValueError("history_steps must be at least 2")
    if data.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError("data contains duplicate storm states")

    feature_rows: list[np.ndarray] = []
    target_rows: list[list[float]] = []
    metadata_rows: list[dict[str, object]] = []
    time_rows: list[list[pd.Timestamp]] = []
    offsets = [pd.Timedelta(hours=6 * step) for step in range(history_steps - 1, -1, -1)]
    future_offset = pd.Timedelta(hours=horizon_hours)

    for sid, storm in data.groupby("SID", sort=False):
        states = storm.sort_values("ISO_TIME").set_index("ISO_TIME", drop=False)
        for issue_time, issue in states.iterrows():
            history_times = [issue_time - offset for offset in offsets]
            target_time = issue_time + future_offset
            if not all(time in states.index for time in history_times) or target_time not in states.index:
                continue
            window = states.loc[history_times]
            target = states.loc[target_time]
            north, east = _displacement_km(
                float(issue.USA_LAT), float(issue.USA_LON), float(target.USA_LAT), float(target.USA_LON)
            )
            feature_rows.append(_window_features(window))
            target_rows.append([north, east, float(target.USA_WIND - issue.USA_WIND)])
            metadata_rows.append({
                "SID": sid,
                "SEASON": int(issue.SEASON),
                "issue_time": issue_time,
                "target_time": target_time,
                "issue_lat": float(issue.USA_LAT),
                "issue_lon": float(issue.USA_LON),
                "issue_wind": float(issue.USA_WIND),
                "target_lat": float(target.USA_LAT),
                "target_lon": float(target.USA_LON),
                "target_wind": float(target.USA_WIND),
            })
            time_rows.append(history_times)

    return ForecastSamples(
        features=np.asarray(feature_rows, dtype=float).reshape((-1, history_steps, len(FEATURE_NAMES))),
        targets=np.asarray(target_rows, dtype=float).reshape((-1, 3)),
        metadata=pd.DataFrame(metadata_rows),
        history_times=np.asarray(time_rows, dtype=object).reshape((-1, history_steps)),
        horizon_hours=horizon_hours,
    )


def storm_split_map(data: pd.DataFrame, train_end_year: int, validation_end_year: int) -> pd.Series:
    """Assign every storm atomically from its first recorded season."""
    if train_end_year >= validation_end_year:
        raise ValueError("train_end_year must be before validation_end_year")
    first_season = data.groupby("SID")["SEASON"].min()
    labels = pd.Series("test", index=first_season.index, dtype="string")
    labels.loc[first_season <= train_end_year] = "train"
    labels.loc[first_season.between(train_end_year + 1, validation_end_year)] = "validation"
    return labels
