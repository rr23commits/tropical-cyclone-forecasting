"""Metadata-only feasibility audit for 3-hourly TCC genesis candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TCC_COLUMNS = ("number", "source", "time", "lat", "lon", "devflag")
IB_COLUMNS = ("SID", "BASIN", "ISO_TIME", "TRACK_TYPE", "LAT", "LON", "USA_WIND")
SPLITS = ((1982, 2007, "train"), (2008, 2011, "validation"), (2012, 2018, "test"))


def load_tcc(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=TCC_COLUMNS)
    data["time"] = pd.to_datetime(data.time, errors="coerce", utc=True)
    for column in ("number", "source", "lat", "lon", "devflag"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def load_ibtracs(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(
        path, skiprows=[1], usecols=IB_COLUMNS, low_memory=False,
        keep_default_na=False, na_values=[" "],
    )
    data["ISO_TIME"] = pd.to_datetime(data.ISO_TIME, errors="coerce", utc=True)
    for column in ("LAT", "LON", "USA_WIND"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _distance_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _wind_covers(times: pd.DatetimeIndex, issue_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    # Tolerate the half-cycle offset between 3-hour TCC and 6-hour IBTrACS timestamps.
    # ponytail: 9h ceiling assumes 6h best-track reports; use per-basin cadence if this is too strict.
    if len(times) < 2:
        return False
    window = times[(times >= issue_time - pd.Timedelta(hours=3)) &
                   (times <= end_time + pd.Timedelta(hours=6))]
    return bool(
        len(window) >= 2 and window[0] <= issue_time + pd.Timedelta(hours=3)
        and window[-1] >= end_time
        and np.diff(window.asi8).max() <= pd.Timedelta(hours=9).value
    )


def _links(tcc: pd.DataFrame, ib: pd.DataFrame) -> tuple[dict[int, tuple[str, str]], dict[str, object]]:
    shared = tcc.merge(ib, left_on="time", right_on="ISO_TIME", suffixes=("_tcc", "_ib"))
    if shared.empty:
        return {}, {"coincident_position_pairs": 0, "ambiguous_tcc_tracks": 0, "single_point_links": 0}
    shared["distance_km"] = _distance_km(
        shared.lat.to_numpy(), shared.lon.to_numpy(), shared.LAT.to_numpy(), shared.LON.to_numpy()
    )
    close = shared.loc[shared.distance_km.le(1000)].copy()
    if close.empty:
        return {}, {"coincident_position_pairs": len(shared), "ambiguous_tcc_tracks": 0, "single_point_links": 0}
    stats = close.groupby(["number", "SID"], as_index=False).agg(
        matched_times=("time", "size"), nearest_km=("distance_km", "min")
    )
    candidate_counts = stats.groupby("number").SID.nunique()
    ambiguous = int(candidate_counts.gt(1).sum())
    # The 1,000-km radius is only a candidate generator; require a unique best
    # overlap/distance score and avoid assigning a SID from a single fix.
    best = stats.sort_values(["number", "matched_times", "nearest_km", "SID"],
                             ascending=[True, False, True, True])
    ranked = best.groupby("number", sort=False)
    top = ranked.head(1).set_index("number")
    runner_up = ranked.nth(1).set_index("number")
    exact_ties = top.index.intersection(runner_up.index[
        top.loc[runner_up.index, "matched_times"].eq(runner_up.matched_times)
        & top.loc[runner_up.index, "nearest_km"].eq(runner_up.nearest_km)
    ])
    best = top.drop(index=exact_ties)
    single_point = best.matched_times.eq(1)
    single_point_count = int(single_point.sum())
    ambiguous_single_points = int(single_point.index[single_point].isin(
        candidate_counts[candidate_counts.gt(1)].index).sum())
    best = best.loc[~single_point]
    nearest = close.sort_values("distance_km").drop_duplicates(["number", "SID"]).set_index(["number", "SID"])
    links = {
        int(number): (str(sid), str(nearest.loc[(number, sid), "BASIN"]))
        for number, sid in best.SID.items()
    }
    return links, {
        "coincident_position_pairs": int(len(shared)),
        "within_1000km_position_pairs": int(len(close)),
        "ambiguous_tcc_tracks": ambiguous,
        "ambiguous_resolved_by_overlap_then_distance": ambiguous - int(len(exact_ties)) - ambiguous_single_points,
        "unresolved_exact_ties": int(len(exact_ties)),
        "ambiguous_single_point_excluded": ambiguous_single_points,
        "single_point_links_excluded": single_point_count,
    }


def build_manifest(tcc_data: pd.DataFrame, ib_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Resolve shared catalogue track IDs, then audit event labels from IBTrACS."""
    if set(TCC_COLUMNS) - set(tcc_data) or set(IB_COLUMNS) - set(ib_data):
        raise ValueError("TCC or IBTrACS input is missing required columns")
    tcc = tcc_data.loc[tcc_data.source.eq(1), list(TCC_COLUMNS)].copy()
    tcc = tcc.dropna(subset=["number", "time", "lat", "lon"])
    tcc["number"] = tcc.number.astype(int)
    if tcc.duplicated(["number", "time"]).any():
        raise ValueError("TCC catalogue has duplicate track/time rows")
    if tcc.groupby("number").devflag.nunique(dropna=False).gt(1).any():
        raise ValueError("TCC devflag must be constant within each track")

    ib = ib_data.loc[ib_data.TRACK_TYPE.eq("main"), list(IB_COLUMNS)].copy()
    ib = ib.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"])
    ib = ib.drop_duplicates(["SID", "ISO_TIME"])
    # The public merged table retains the same catalogue number on a TCC track
    # and its source=2 IB-derived continuation. Use that identifier first; only
    # map the continuation to NOAA IBTrACS by coincident positions afterward.
    # Source=2 is label-linkage metadata; manifest inputs remain source=1 TCC rows.
    bridge = tcc_data.loc[tcc_data.source.eq(2) & tcc_data.number.isin(tcc.number),
                          ["number", "source", "time", "lat", "lon"]].copy()
    bridge = bridge.dropna(subset=["number", "time", "lat", "lon"])
    tcc_edges = tcc.sort_values("time").groupby("number", as_index=False).tail(1).set_index("number")
    bridge_edges = bridge.sort_values("time").groupby("number", as_index=False).head(1).set_index("number")
    shared_ids = tcc_edges.index.intersection(bridge_edges.index)
    contiguous_ids = shared_ids[
        bridge_edges.loc[shared_ids, "time"].eq(tcc_edges.loc[shared_ids, "time"] + pd.Timedelta(hours=3))
        & (_distance_km(
            tcc_edges.loc[shared_ids, "lat"].to_numpy(), tcc_edges.loc[shared_ids, "lon"].to_numpy(),
            bridge_edges.loc[shared_ids, "lat"].to_numpy(), bridge_edges.loc[shared_ids, "lon"].to_numpy(),
        ) <= 1000)
    ]
    bridge = bridge.loc[bridge.number.isin(contiguous_ids)].copy()
    links, diagnostics = _links(bridge, ib)
    track_flags = tcc.groupby("number").devflag.first()
    diagnostics["tcc_tracks_with_shared_bridge_id"] = int(len(shared_ids))
    diagnostics["validated_3h_bridge_ids"] = int(len(contiguous_ids))
    diagnostics["developing_flag_tracks"] = int(track_flags.eq(1).sum())
    diagnostics["developing_flag_tracks_with_bridge"] = int(track_flags[track_flags.eq(1)].index.isin(contiguous_ids).sum())
    crossing_rows = ib.loc[ib.USA_WIND.ge(34)].groupby("SID").ISO_TIME.min()
    sid_wind_times = {
        str(sid): pd.DatetimeIndex(group.ISO_TIME.sort_values())
        for sid, group in ib.dropna(subset=["USA_WIND"]).groupby("SID", sort=False)
    }
    ib_sids = set(ib.loc[ib.ISO_TIME.dt.year.between(1982, 2018), "SID"].astype(str))
    linked_sids = {sid for sid, _ in links.values()}

    rows: list[dict[str, object]] = []
    counts = {"tracks": int(tcc.number.nunique()), "linked_tcc_tracks": len(links),
              "unlinked_tcc_tracks": int(tcc.number.nunique() - len(links)),
              "ibtracs_sids_1982_2018": len(ib_sids), "linked_ibtracs_sids": len(linked_sids),
              "unlinked_ibtracs_sids": len(ib_sids - linked_sids), **diagnostics,
              "insufficient_history": 0, "already_genesis": 0, "insufficient_24h_followup": 0,
              "split_range_or_boundary_purged": 0, "unresolved_developing_tracks": 0,
              "unknown_basin_tracks": 0, "unknown_basin_flagged_nondev": 0}

    for number, track in tcc.sort_values("time").groupby("number", sort=False):
        track = track.set_index("time", drop=False)
        times = set(track.index)
        start_year = int(track.index.min().year)
        split_row = next((item for item in SPLITS if item[0] <= start_year <= item[1]), None)
        if split_row is None:
            counts["split_range_or_boundary_purged"] += len(track)
            continue
        first_year, last_year, split = split_row
        split_end = pd.Timestamp(year=last_year + 1, month=1, day=1, tz="UTC")
        sid, basin = links.get(int(number), ("", "UNKNOWN"))
        catalogue_nondev = bool(track.devflag.iloc[0] == 0)
        if not sid and not catalogue_nondev:
            counts["unresolved_developing_tracks"] += 1
            continue
        if not sid:
            counts["unknown_basin_tracks"] += 1
            counts["unknown_basin_flagged_nondev"] += 1
        genesis = crossing_rows.get(sid, pd.NaT) if sid else pd.NaT
        wind_times = sid_wind_times.get(sid, pd.DatetimeIndex([]))
        for issue_time, issue in track.iterrows():
            history = [issue_time - pd.Timedelta(hours=24 - 3 * i) for i in range(9)]
            if not all(stamp in times for stamp in history):
                counts["insufficient_history"] += 1
                continue
            if not first_year <= issue_time.year <= last_year:
                counts["split_range_or_boundary_purged"] += 1
                continue
            if pd.notna(genesis) and genesis <= issue_time:
                counts["already_genesis"] += 1
                continue
            end24 = issue_time + pd.Timedelta(hours=24)
            if end24 >= split_end:
                counts["split_range_or_boundary_purged"] += 1
                continue
            if pd.notna(genesis) and issue_time < genesis <= end24:
                label24: int | None = 1
            elif (sid and _wind_covers(wind_times, issue_time, end24)) or (
                    not sid and catalogue_nondev and
                    all(issue_time + pd.Timedelta(hours=h) in times for h in range(3, 25, 3))):
                label24 = 0
            else:
                counts["insufficient_24h_followup"] += 1
                continue

            end48 = issue_time + pd.Timedelta(hours=48)
            label48: int | None = None
            if end48 < split_end:
                if pd.notna(genesis) and issue_time < genesis <= end48:
                    label48 = 1
                elif (sid and _wind_covers(wind_times, issue_time, end48)) or (
                        not sid and catalogue_nondev and
                        all(issue_time + pd.Timedelta(hours=h) in times for h in range(3, 49, 3))):
                    label48 = 0
            rows.append({
                "tcc_track_id": int(number), "basin": basin, "year": int(issue_time.year),
                "issue_time_utc": issue_time.isoformat().replace("+00:00", "Z"),
                "current_lat": float(issue.lat), "current_lon": float(issue.lon),
                "linked_ibtracs_sid": sid or "",
                "first_34kt_time_utc": "" if pd.isna(genesis) else genesis.isoformat().replace("+00:00", "Z"),
                "label_24h": label24, "label_48h": label48,
                "input_times_utc": "|".join(stamp.isoformat().replace("+00:00", "Z") for stamp in history),
                "split": split,
            })

    manifest = pd.DataFrame(rows, columns=[
        "tcc_track_id", "basin", "year", "issue_time_utc", "current_lat", "current_lon",
        "linked_ibtracs_sid", "first_34kt_time_utc", "label_24h", "label_48h",
        "input_times_utc", "split",
    ])
    counts["eligible_issue_times"] = len(manifest)
    counts["positive_24h"] = int(manifest.label_24h.eq(1).sum())
    counts["negative_24h"] = int(manifest.label_24h.eq(0).sum())
    counts["positive_48h"] = int(manifest.label_48h.eq(1).sum())
    counts["negative_48h"] = int(manifest.label_48h.eq(0).sum())
    counts["unlabeled_48h"] = int(manifest.label_48h.isna().sum())
    basin_resolved = manifest.loc[manifest.basin.ne("UNKNOWN")]
    counts["basin_resolved_issue_times"] = len(basin_resolved)
    counts["basin_resolved_positive_24h"] = int(basin_resolved.label_24h.eq(1).sum())
    counts["basin_resolved_negative_24h"] = int(basin_resolved.label_24h.eq(0).sum())
    return manifest, counts


def report(manifest: pd.DataFrame, counts: dict[str, object]) -> None:
    print(json.dumps(counts, indent=2, default=int))
    if manifest.empty:
        return
    summary = manifest.assign(
        positive_24h=manifest.label_24h.eq(1), negative_24h=manifest.label_24h.eq(0),
        positive_48h=manifest.label_48h.eq(1), negative_48h=manifest.label_48h.eq(0),
        unlabeled_48h=manifest.label_48h.isna(),
    )
    print("\nBy split:")
    print(summary.groupby("split").agg(samples=("tcc_track_id", "size"), positive_24h=("positive_24h", "sum"),
                                        negative_24h=("negative_24h", "sum"), positive_48h=("positive_48h", "sum"),
                                        negative_48h=("negative_48h", "sum"), unlabeled_48h=("unlabeled_48h", "sum")).to_string())
    print("\nBasin-resolved subset by split (exclude basin=UNKNOWN):")
    resolved = summary.loc[summary.basin.ne("UNKNOWN")]
    print(resolved.groupby("split").agg(samples=("tcc_track_id", "size"), positive_24h=("positive_24h", "sum"),
                                          negative_24h=("negative_24h", "sum"), positive_48h=("positive_48h", "sum"),
                                          negative_48h=("negative_48h", "sum"), unlabeled_48h=("unlabeled_48h", "sum")).to_string())
    print("\nBy basin and issue year:")
    print(summary.groupby(["basin", "year"]).agg(samples=("tcc_track_id", "size"), positive_24h=("positive_24h", "sum"),
                                                   negative_24h=("negative_24h", "sum")).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tcc_csv", type=Path, help="public TCC+IBTrACS metadata CSV (source=1 TCC tracks)")
    parser.add_argument("ibtracs_csv", type=Path, help="official global IBTrACS v04r01 CSV")
    parser.add_argument("--output", type=Path, required=True, help="candidate manifest CSV output path")
    args = parser.parse_args()
    manifest, counts = build_manifest(load_tcc(args.tcc_csv), load_ibtracs(args.ibtracs_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    basin_year_path = args.output.parent / "genesis_counts_by_basin_year.csv"
    yearly = manifest.assign(
        positive_24h=manifest.label_24h.eq(1), negative_24h=manifest.label_24h.eq(0),
        positive_48h=manifest.label_48h.eq(1), negative_48h=manifest.label_48h.eq(0),
        unlabeled_48h=manifest.label_48h.isna(),
    )
    yearly.groupby(["basin", "year"]).agg(
        samples=("tcc_track_id", "size"), tracks=("tcc_track_id", "nunique"),
        positive_24h=("positive_24h", "sum"), negative_24h=("negative_24h", "sum"),
        positive_48h=("positive_48h", "sum"), negative_48h=("negative_48h", "sum"),
        unlabeled_48h=("unlabeled_48h", "sum"),
    ).reset_index().to_csv(basin_year_path, index=False)
    report(manifest, counts)
    print(f"\nManifest: {args.output}")
    print(f"Basin/year counts: {basin_year_path}")


if __name__ == "__main__":
    main()
