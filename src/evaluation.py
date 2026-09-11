"""Reconstruct predicted states and score direct tropical-cyclone forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ibtracs import ForecastSamples


EARTH_RADIUS_KM = 6371.0088


def reconstruct_states(samples: ForecastSamples, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert local north/east displacement and wind-change predictions to states."""
    if predictions.shape != (len(samples.metadata), 3):
        raise ValueError("predictions must have shape (number of samples, 3)")
    issue_lat = samples.metadata["issue_lat"].to_numpy(dtype=float)
    issue_lon = samples.metadata["issue_lon"].to_numpy(dtype=float)
    latitude = issue_lat + predictions[:, 0] / 111.32
    mean_latitude = np.deg2rad((issue_lat + latitude) / 2.0)
    longitude = issue_lon + predictions[:, 1] / (111.32 * np.cos(mean_latitude))
    longitude = (longitude + 180.0) % 360.0 - 180.0
    wind = samples.metadata["issue_wind"].to_numpy(dtype=float) + predictions[:, 2]
    return latitude, longitude, wind


def haversine_km(
    latitude_a: np.ndarray, longitude_a: np.ndarray, latitude_b: np.ndarray, longitude_b: np.ndarray
) -> np.ndarray:
    """Return great-circle distances in kilometres for paired coordinates."""
    lat_a, lon_a, lat_b, lon_b = map(np.deg2rad, (latitude_a, longitude_a, latitude_b, longitude_b))
    haversine = np.sin((lat_b - lat_a) / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))


def evaluate(samples: ForecastSamples, predictions: np.ndarray) -> dict[str, float | int]:
    """Score a model on one horizon's already-selected forecast origins."""
    predicted_lat, predicted_lon, predicted_wind = reconstruct_states(samples, predictions)
    target_lat = samples.metadata["target_lat"].to_numpy(dtype=float)
    target_lon = samples.metadata["target_lon"].to_numpy(dtype=float)
    target_wind = samples.metadata["target_wind"].to_numpy(dtype=float)
    track_errors = haversine_km(predicted_lat, predicted_lon, target_lat, target_lon)
    wind_errors = predicted_wind - target_wind
    return {
        "origins": len(samples.metadata),
        "storms": int(samples.metadata["SID"].nunique()),
        "track_error_km": float(track_errors.mean()),
        "wind_mae": float(np.abs(wind_errors).mean()),
        "wind_rmse": float(np.sqrt(np.mean(wind_errors ** 2))),
    }


def subset_samples(samples: ForecastSamples, indices: np.ndarray) -> ForecastSamples:
    """Select a split without changing its feature, target, or timing alignment."""
    return ForecastSamples(
        features=samples.features[indices],
        targets=samples.targets[indices],
        metadata=samples.metadata.iloc[indices].reset_index(drop=True),
        history_times=samples.history_times[indices],
        horizon_hours=samples.horizon_hours,
    )


def split_samples(samples: ForecastSamples, split_map: pd.Series) -> dict[str, ForecastSamples]:
    """Split samples by the precomputed atomic storm labels."""
    labels = samples.metadata["SID"].map(split_map)
    if labels.isna().any():
        raise ValueError("every sample storm must have a split label")
    return {
        split: subset_samples(samples, np.flatnonzero(labels.eq(split).to_numpy()))
        for split in ("train", "validation", "test")
    }


def align_common_origins(sample_sets: list[ForecastSamples]) -> list[ForecastSamples]:
    """Restrict sample sets to the same storm issue/target times for a fair comparison."""
    if not sample_sets:
        return []
    key_columns = ["SID", "issue_time", "target_time"]
    indexes = []
    for samples in sample_sets:
        index = pd.MultiIndex.from_frame(samples.metadata[key_columns])
        if not index.is_unique:
            raise ValueError("sample origins must be unique")
        indexes.append(index)
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    if common.empty:
        raise ValueError("history lengths have no common forecast origins")
    return [subset_samples(samples, np.flatnonzero(index.isin(common))) for samples, index in zip(sample_sets, indexes)]
