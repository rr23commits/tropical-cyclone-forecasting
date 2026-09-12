#!/usr/bin/env python3
"""Export existing held-out errors as a small static research-replay payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = ("persistence", "constant-motion", "ridge", "gru")


def _point(row: pd.Series, prefix: str = "") -> dict[str, float | str]:
    """Serialize a measured or predicted state without adding derived claims."""
    return {
        "time": str(row[f"{prefix}time"]),
        "lat": float(row[f"{prefix}lat"]),
        "lon": float(row[f"{prefix}lon"]),
        "wind": float(row[f"{prefix}wind"]),
    }


def load_storm_identity(path: Path) -> dict[str, dict[str, object]]:
    """Read only real display identity fields; experiment labels remain untouched."""
    if not path.exists():
        return {}
    identity = pd.read_csv(
        path, skiprows=[1], usecols=["SID", "SEASON", "NAME", "BASIN", "USA_ATCF_ID"],
        keep_default_na=False, na_values=[" "], low_memory=False,
    )
    identity = identity.sort_values("SID").groupby("SID", as_index=False).first()
    return {
        row.SID: {
            "name": row.NAME or "UNNAMED", "basin": row.BASIN or "NA",
            "stormId": row.USA_ATCF_ID or row.SID, "season": int(row.SEASON),
        }
        for row in identity.itertuples(index=False)
    }


def build_payload(
    rows: pd.DataFrame, metrics: pd.DataFrame, identities: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Group the frozen, held-out rows into observed tracks and forecast issue times."""
    required = {
        "SID", "SEASON", "issue_time", "target_time", "issue_lat", "issue_lon", "issue_wind",
        "target_lat", "target_lon", "target_wind", "model", "horizon_hours", "predicted_lat",
        "predicted_lon", "predicted_wind", "track_error_km", "wind_abs_error_kt",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"replay export is missing columns: {sorted(missing)}")

    identities = identities or {}
    storms = []
    for sid, storm_rows in rows.groupby("SID", sort=True):
        # All models repeat the same observed source/target states; retain one exact copy.
        observed = {}
        for _, row in storm_rows.drop_duplicates(["issue_time"]).iterrows():
            observed[str(row.issue_time)] = _point(row, "issue_")
        for _, row in storm_rows.drop_duplicates(["target_time"]).iterrows():
            observed[str(row.target_time)] = _point(row, "target_")

        forecasts = []
        for (issue_time, horizon), group in storm_rows.groupby(["issue_time", "horizon_hours"], sort=True):
            row = group.iloc[0]
            predictions = {
                str(prediction.model): {
                    "lat": float(prediction.predicted_lat), "lon": float(prediction.predicted_lon),
                    "wind": float(prediction.predicted_wind), "trackErrorKm": float(prediction.track_error_km),
                    "windAbsErrorKt": float(prediction.wind_abs_error_kt),
                }
                for _, prediction in group.iterrows()
            }
            forecasts.append({
                "issueTime": str(issue_time), "horizonHours": int(horizon),
                "actual": _point(row, "target_"), "predictions": predictions,
            })
        identity = identities.get(sid, {})
        storms.append({
            "sid": sid, "season": int(identity.get("season", storm_rows.SEASON.iloc[0])),
            "name": identity.get("name", "UNNAMED"), "basin": identity.get("basin", "NA"),
            "stormId": identity.get("stormId", sid),
            "observed": sorted(observed.values(), key=lambda point: point["time"]), "forecasts": forecasts,
        })

    summary = metrics.loc[metrics.model.isin(MODELS), [
        "model", "horizon_hours", "origins", "storms", "track_mean_km", "wind_mae_kt",
    ]].to_dict("records")
    return {
        "label": "RESEARCH REPLAY / NON-OPERATIONAL",
        "source": "IBTrACS v04r01 North Atlantic / hurdat_atl held-out test storms (2020–2025)",
        "models": list(MODELS), "horizons": [6, 12, 24, 48], "storms": storms, "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, default=Path("results/track_only_error_analysis/held_out_errors.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("results/track_only_error_analysis/overall.csv"))
    parser.add_argument("--ibtracs", type=Path, default=Path("/private/tmp/ibtracs.NA.list.v04r01.csv"), help="optional official source CSV for storm display identity")
    parser.add_argument("--output", type=Path, default=Path("frontend/data/replay.json"))
    args = parser.parse_args()
    payload = build_payload(pd.read_csv(args.errors), pd.read_csv(args.metrics), load_storm_identity(args.ibtracs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(payload['storms'])} held-out storms to {args.output}")


if __name__ == "__main__":
    main()
