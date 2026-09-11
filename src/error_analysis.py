"""Held-out forecast-error rows and compact, reproducible summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import haversine_km, reconstruct_states
from .ibtracs import ForecastSamples


def intensity_bin(wind_kt: pd.Series) -> pd.Series:
    """Use descriptive issue-intensity groups; this is analysis, not a new label source."""
    return pd.cut(
        wind_kt, [-np.inf, 33, 63, 82, 95, 112, np.inf],
        labels=["TD (<34)", "TS (34-63)", "Cat 1 (64-82)", "Cat 2 (83-95)", "Cat 3 (96-112)", "Major (>=113)"],
    ).astype("string")


def motion_bin(speed_kmh: pd.Series) -> pd.Series:
    """Separate slow, typical, and fast realised storm motion for descriptive checks."""
    return pd.cut(speed_kmh, [-np.inf, 10, 25, np.inf], labels=["slow (<10)", "typical (10-25)", "fast (>=25)"]).astype("string")


def wind_change_bin(change_kt: pd.Series) -> pd.Series:
    """Describe realised target change without extrapolating a lead-specific rate."""
    return pd.cut(change_kt, [-np.inf, -5, 5, np.inf], labels=["weakening (<-5)", "steady (-5 to 5)", "strengthening (>5)"]).astype("string")


def error_rows(samples: ForecastSamples, prediction: np.ndarray, model: str) -> pd.DataFrame:
    """Return one independently auditable held-out error row per forecast origin."""
    predicted_lat, predicted_lon, predicted_wind = reconstruct_states(samples, prediction)
    rows = samples.metadata.copy()
    target_lat = rows["target_lat"].to_numpy(dtype=float)
    target_lon = rows["target_lon"].to_numpy(dtype=float)
    issue_lat = rows["issue_lat"].to_numpy(dtype=float)
    issue_lon = rows["issue_lon"].to_numpy(dtype=float)
    rows["model"] = model
    rows["horizon_hours"] = samples.horizon_hours
    rows["predicted_lat"] = predicted_lat
    rows["predicted_lon"] = predicted_lon
    rows["predicted_wind"] = predicted_wind
    rows["track_error_km"] = haversine_km(predicted_lat, predicted_lon, target_lat, target_lon)
    rows["wind_error_kt"] = predicted_wind - rows["target_wind"].to_numpy(dtype=float)
    rows["wind_abs_error_kt"] = rows["wind_error_kt"].abs()
    rows["target_motion_kmh"] = haversine_km(issue_lat, issue_lon, target_lat, target_lon) / samples.horizon_hours
    rows["issue_intensity_bin"] = intensity_bin(rows["issue_wind"])
    rows["motion_bin"] = motion_bin(rows["target_motion_kmh"])
    rows["wind_change_bin"] = wind_change_bin(rows["target_wind"] - rows["issue_wind"])
    return rows


def summarize(rows: pd.DataFrame, by: list[str], bootstrap_samples: int = 2000) -> pd.DataFrame:
    """Summarize a specified grouping without treating correlated storm rows as independent storms."""
    grouped = rows.groupby(by, observed=True, dropna=False)
    summary = grouped.agg(
        origins=("SID", "size"),
        storms=("SID", "nunique"),
        track_mean_km=("track_error_km", "mean"),
        track_median_km=("track_error_km", "median"),
        track_p90_km=("track_error_km", lambda values: values.quantile(0.90)),
        track_p95_km=("track_error_km", lambda values: values.quantile(0.95)),
        wind_mae_kt=("wind_abs_error_kt", "mean"),
        wind_rmse_kt=("wind_error_kt", lambda values: float(np.sqrt(np.mean(values ** 2)))),
        wind_bias_kt=("wind_error_kt", "mean"),
    ).reset_index()
    # Storm windows are serially correlated.  Report a storm-macro bootstrap interval
    # rather than an origin-level interval that would overstate precision.
    intervals: list[dict[str, object]] = []
    random = np.random.default_rng(42)
    for key, group in grouped:
        values = group.groupby("SID", observed=True).agg(
            track_error_km=("track_error_km", "mean"), wind_abs_error_kt=("wind_abs_error_kt", "mean")
        )
        indices = random.integers(0, len(values), size=(bootstrap_samples, len(values)))
        track_draws = values["track_error_km"].to_numpy()[indices].mean(axis=1)
        wind_draws = values["wind_abs_error_kt"].to_numpy()[indices].mean(axis=1)
        keys = key if isinstance(key, tuple) else (key,)
        intervals.append(dict(zip(by, keys)) | {
            "storm_macro_track_mean_km": float(values["track_error_km"].mean()),
            "storm_macro_wind_mae_kt": float(values["wind_abs_error_kt"].mean()),
            "storm_track_ci95_low_km": float(np.quantile(track_draws, 0.025)),
            "storm_track_ci95_high_km": float(np.quantile(track_draws, 0.975)),
            "storm_wind_mae_ci95_low_kt": float(np.quantile(wind_draws, 0.025)),
            "storm_wind_mae_ci95_high_kt": float(np.quantile(wind_draws, 0.975)),
        })
    return summary.merge(pd.DataFrame(intervals), on=by).sort_values(by).reset_index(drop=True)
