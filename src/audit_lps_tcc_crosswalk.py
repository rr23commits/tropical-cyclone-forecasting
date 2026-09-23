"""Metadata-only, one-to-one crosswalk of frozen TCC histories to LPS detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_genesis import _distance_km, load_tcc
from src.preflight_genesis import cohort


DISTANCE_LIMIT_KM = 6371.0 * np.deg2rad(1.0)
LPS_COLUMNS = (
    "row_id", "track_id", "time", "lat_detected", "lon_detected", "position_source",
    "mean_vort_850_detectedcentre", "rh700_mean_pct_detectedcentre",
)
FEATURE_COLUMNS = ("mean_vort_850_detectedcentre", "rh700_mean_pct_detectedcentre")


def read_lps_catalogue(path: Path) -> pd.DataFrame:
    """Read only position/identity/feature columns; parquet needs an installed engine."""
    if path.suffix.lower() == ".parquet":
        try:
            data = pd.read_parquet(path, columns=list(LPS_COLUMNS))
        except ImportError as error:
            raise RuntimeError("reading the Parquet catalogue requires pyarrow or fastparquet") from error
    else:
        data = pd.read_csv(path, usecols=list(LPS_COLUMNS), low_memory=False)
    missing = set(LPS_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"LPS catalogue lacks required columns: {sorted(missing)}")
    data["time"] = pd.to_datetime(data.time, errors="coerce", utc=True)
    for col in ("row_id", "track_id", "lat_detected", "lon_detected", *FEATURE_COLUMNS):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def _detected_rows(lps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = lps.copy()
    source = data.position_source.fillna("").astype(str).str.lower()
    observed = (
        data.time.notna() & data.lat_detected.notna() & data.lon_detected.notna()
        & source.str.contains("observ|detect", regex=True)
        & ~source.str.contains("posterior|interpol|gap|smooth|project|final", regex=True)
    )
    detected = data.loc[observed].copy()
    if detected.row_id.isna().any() or detected.duplicated(["time", "row_id"]).any():
        raise ValueError("observed LPS detections need unique, non-null (time, row_id) keys")
    return detected, data.loc[~observed & data.time.notna()].copy()


def build_crosswalk(manifest: pd.DataFrame, tcc_data: pd.DataFrame, lps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return issue-row and unique TCC-position audits without resolving ambiguity silently."""
    rows = manifest.reset_index(drop=True).copy()
    lps = lps.copy()
    lps["time"] = pd.to_datetime(lps.time, errors="coerce", utc=True)
    required = {"tcc_track_id", "issue_time_utc", "input_times_utc", "current_lat", "current_lon",
                "basin", "label_24h", "label_48h", "split"}
    if required - set(rows):
        raise ValueError(f"TCC manifest lacks {sorted(required - set(rows))}")

    tcc = tcc_data.loc[tcc_data.source.eq(1)].copy()
    tcc["number"] = tcc.number.astype(int)
    if tcc.duplicated(["number", "time"]).any():
        raise ValueError("source=1 TCC has duplicate (track, time) positions")
    tcc_positions = tcc.set_index(["number", "time"])
    refs: list[dict] = []
    for row_no, row in rows.iterrows():
        stamps = pd.to_datetime(str(row.input_times_utc).split("|"), errors="coerce", utc=True)
        if len(stamps) != 9 or stamps.isna().any():
            raise ValueError(f"TCC row {row_no} does not have nine valid UTC input times")
        if any(stamps[i] - stamps[i - 1] != pd.Timedelta(hours=3) for i in range(1, 9)):
            raise ValueError(f"TCC row {row_no} input times are not exact 3-hour steps")
        if stamps[-1] != pd.Timestamp(row.issue_time_utc):
            raise ValueError(f"TCC row {row_no} issue time differs from its final history time")
        for step, stamp in enumerate(stamps):
            key = (int(row.tcc_track_id), stamp)
            if key not in tcc_positions.index:
                raise ValueError(f"missing source=1 TCC position for track/time {key}")
            point = tcc_positions.loc[key]
            if step == 8 and not np.allclose(
                [point.lat, point.lon], [float(row.current_lat), float(row.current_lon)], rtol=0, atol=1e-6
            ):
                raise ValueError(f"manifest issue position disagrees with source=1 TCC for {key}")
            refs.append({"manifest_row": row_no, "step": step, "tcc_track_id": key[0], "time": stamp,
                         "tcc_lat": float(point.lat), "tcc_lon": float(point.lon)})

    # Repeated history states across overlapping issue windows are one physical
    # TCC position, so match each (track,time) once and then reuse that audit row.
    unique = pd.DataFrame(refs).drop_duplicates(["tcc_track_id", "time"]).reset_index(drop=True)
    detected, excluded = _detected_rows(lps)
    lps_by_time = {time: group.reset_index(drop=True) for time, group in detected.groupby("time", sort=False)}
    per_time: dict[pd.Timestamp, list[dict]] = {}
    for stamp, group in unique.groupby("time", sort=False):
        candidates = lps_by_time.get(stamp, detected.iloc[0:0])
        if candidates.empty:
            per_time[stamp] = [{"distances": np.array([]), "candidate_indexes": np.array([], dtype=int)} for _ in group.index]
            continue
        dists = _distance_km(
            group.tcc_lat.to_numpy()[:, None], group.tcc_lon.to_numpy()[:, None],
            candidates.lat_detected.to_numpy()[None, :], candidates.lon_detected.to_numpy()[None, :],
        )
        per_time[stamp] = [
            {"distances": dists[i], "candidate_indexes": np.flatnonzero(dists[i] <= DISTANCE_LIMIT_KM)}
            for i in range(len(group))
        ]

    assignments: list[dict] = []
    position_offsets = {}
    for stamp, group in unique.groupby("time", sort=False):
        position_offsets.update({int(index): offset for offset, index in enumerate(group.index)})
    # Count reverse claims first; shared detections are ambiguous for every claimant.
    reverse_claims: dict[tuple[pd.Timestamp, int], int] = {}
    for stamp, specs in per_time.items():
        for spec in specs:
            candidates = lps_by_time.get(stamp, detected.iloc[0:0])
            for idx in spec["candidate_indexes"]:
                reverse_claims[(stamp, int(candidates.iloc[idx].row_id))] = reverse_claims.get((stamp, int(candidates.iloc[idx].row_id)), 0) + 1

    for i, point in unique.iterrows():
        stamp = point.time
        candidates = lps_by_time.get(stamp, detected.iloc[0:0])
        spec = per_time[stamp][position_offsets[i]]
        distances = spec["distances"]
        nearby = spec["candidate_indexes"]
        nearest_idx = int(np.argmin(distances)) if len(distances) else None
        nearest_distance = float(distances[nearest_idx]) if nearest_idx is not None else None
        nearest = candidates.iloc[nearest_idx] if nearest_idx is not None else None
        collision = any(reverse_claims.get((stamp, int(candidates.iloc[idx].row_id)), 0) > 1 for idx in nearby)
        accepted = len(nearby) == 1 and not collision
        if accepted:
            lps_row = candidates.iloc[int(nearby[0])]
            status = "matched"
        elif len(nearby) > 1:
            lps_row = None
            status = "ambiguous_multiple_lps_candidates"
        elif collision:
            lps_row = None
            status = "ambiguous_shared_lps_detection"
        else:
            lps_row = None
            status = "no_candidate_within_1deg"
        assignments.append({
            "tcc_track_id": int(point.tcc_track_id), "time_utc": stamp.isoformat().replace("+00:00", "Z"),
            "tcc_lat": float(point.tcc_lat), "tcc_lon": float(point.tcc_lon),
            "lps_rows_at_time": int((lps.time == stamp).sum()), "lps_detected_rows_at_time": len(candidates),
            "non_detected_rows_at_time": int((excluded.time == stamp).sum()),
            "nearest_lps_distance_km": nearest_distance,
            "nearest_lps_row_id": None if nearest is None else int(nearest.row_id),
            "nearest_lps_track_id": None if nearest is None else int(nearest.track_id),
            "candidate_count_within_1deg": len(nearby), "ambiguous": status.startswith("ambiguous"),
            "match_status": status, "matched": bool(accepted),
            "matched_lps_row_id": None if lps_row is None else int(lps_row.row_id),
            "matched_lps_track_id": None if lps_row is None else int(lps_row.track_id),
            "accepted_distance_km": None if lps_row is None else float(distances[int(nearby[0])]),
            "position_source": None if lps_row is None else str(lps_row.position_source),
            **{f"{feature}_value": None if lps_row is None or pd.isna(lps_row[feature]) else float(lps_row[feature])
               for feature in FEATURE_COLUMNS},
        })
    positions = pd.DataFrame(assignments)
    match_lookup = {(int(r.tcc_track_id), pd.Timestamp(r.time_utc)): r for r in positions.itertuples(index=False)}

    expanded = []
    for ref in refs:
        match = match_lookup[(ref["tcc_track_id"], ref["time"])]
        expanded.append({"manifest_row": ref["manifest_row"], "step": ref["step"], "matched": match.matched,
                         "match_status": match.match_status, "matched_lps_track_id": match.matched_lps_track_id,
                         **{f"{feature}_value": getattr(match, f"{feature}_value") for feature in FEATURE_COLUMNS}})
    expanded = pd.DataFrame(expanded)
    grouped = expanded.groupby("manifest_row", sort=False)
    rows["matched_timestamps"] = grouped.matched.sum().reindex(rows.index, fill_value=0).astype(int)
    rows["full_9_step_match"] = rows.matched_timestamps.eq(9)
    rows["match_class"] = np.where(rows.matched_timestamps.eq(0), "zero",
                                   np.where(rows.full_9_step_match, "full_9_of_9", "partial_1_to_8"))

    full_rows = rows.loc[rows.full_9_step_match]
    full_expanded = expanded.loc[expanded.manifest_row.isin(full_rows.index)]
    feature_coverage = {}
    for feature in FEATURE_COLUMNS:
        values = pd.to_numeric(full_expanded[f"{feature}_value"], errors="coerce")
        counts = values.notna().groupby(full_expanded.manifest_row).sum().reindex(full_rows.index, fill_value=0)
        feature_coverage[feature] = {
            "non_null_timestamp_references": int(values.notna().sum()),
            "rows_with_9_of_9": int(counts.eq(9).sum()),
            "rows_with_0_to_9_counts": {str(n): int(v) for n, v in counts.value_counts().sort_index().items()},
        }
    both_complete = 0
    for row_no in full_rows.index:
        sample = expanded.loc[expanded.manifest_row.eq(row_no)]
        both_complete += int(all(pd.to_numeric(sample[f"{f}_value"], errors="coerce").notna().all() for f in FEATURE_COLUMNS))

    accepted_positions = positions.loc[positions.matched]
    required_times = set(unique.time)
    excluded_at_required_times = int(excluded.time.isin(required_times).sum())
    distances = accepted_positions.accepted_distance_km.astype(float)
    full_track_use: dict[int, set[int]] = {}
    for row_no in full_rows.index:
        tracks = set(expanded.loc[expanded.manifest_row.eq(row_no), "matched_lps_track_id"].dropna().astype(int))
        for track_id in tracks:
            full_track_use.setdefault(track_id, set()).add(int(row_no))
    track_rows = sorted(((track, len(row_ids)) for track, row_ids in full_track_use.items()), key=lambda x: (-x[1], x[0]))
    largest_track_rows = track_rows[0][1] if track_rows else 0
    full_count = len(full_rows)

    def group_rates(column: str) -> list[dict]:
        result = []
        for key, group in rows.groupby(column, dropna=False, sort=True):
            result.append({"group": "" if pd.isna(key) else str(key), "rows": len(group),
                           "full_9_step_rows": int(group.full_9_step_match.sum()),
                           "full_9_step_percent": round(100 * group.full_9_step_match.mean(), 3)})
        return result

    distribution = rows.matched_timestamps.value_counts().reindex(range(10), fill_value=0)
    class_split = {
        f"{split}:{label}": int(len(group))
        for (split, label), group in full_rows.groupby(["split", "label_24h"], dropna=False)
    }
    label_counts = {str(label): int(count) for label, count in full_rows.label_24h.value_counts(dropna=False).items()}
    full_track_count = int(full_rows.tcc_track_id.nunique())
    both_complete_rate = both_complete / full_count if full_count else 0.0
    split_classes_present = all(class_split.get(f"{split}:{label}", 0) >= 10
                                for split in ("train", "validation", "test") for label in (0, 1))
    # A fixed screening heuristic makes the eventual A/B distinction reproducible.
    semester_ready = (
        full_count >= 100 and full_track_count >= 20
        and min(label_counts.get("0", 0), label_counts.get("1", 0)) >= 20
        and split_classes_present and both_complete_rate >= 0.90
        and not (full_count and 100 * largest_track_rows / full_count > 10)
    )
    verdict = "A" if semester_ready else ("C" if full_count == 0 else "B")
    summary = {
        "status": "complete", "distance_threshold_degrees": 1.0, "distance_threshold_km": DISTANCE_LIMIT_KM,
        "matching_rule": "same UTC timestamp; accept only a unique within-threshold LPS detected center that is not claimed by another unique TCC track-time position",
        "tcc_rows_examined": len(rows), "zero_matches": int(rows.matched_timestamps.eq(0).sum()),
        "partial_1_to_8_matches": int(rows.matched_timestamps.between(1, 8).sum()),
        "full_9_of_9_matches": full_count,
        "full_match_percent": round(100 * full_count / len(rows), 3) if len(rows) else 0.0,
        "full_matches_by_basin": group_rates("basin"), "full_matches_by_split": group_rates("split"),
        "full_matches_by_label_24h": group_rates("label_24h"),
        "full_matches_by_basin_year_split": [
            {"basin": str(b), "year": int(y), "split": str(s), "rows": len(g), "full_9_step_rows": int(g.full_9_step_match.sum())}
            for (b, y, s), g in rows.groupby(["basin", "year", "split"], sort=True)
        ],
        "retained_subset": {"full_9_of_9_rows": full_count, "unique_tcc_tracks": full_track_count,
                            "years": sorted(int(x) for x in full_rows.year.unique()),
                            "split_label_counts": class_split},
        "matched_timestamp_distribution": {str(i): int(n) for i, n in distribution.items()},
        "accepted_unique_position_distance_km": {
            "count": int(distances.notna().sum()),
            "median": None if distances.empty else float(distances.median()),
            "mean": None if distances.empty else float(distances.mean()),
            "p90": None if distances.empty else float(np.percentile(distances, 90)),
            "maximum": None if distances.empty else float(distances.max()),
        },
        "ambiguous_unique_tcc_positions": int(positions.ambiguous.sum()),
        "matches_using_non_detected_or_interpolated_rows": 0,
        "non_detected_lps_rows_at_required_timestamps_excluded": excluded_at_required_times,
        "unique_lps_track_ids_used": int(accepted_positions.matched_lps_track_id.nunique()),
        "lps_track_concentration": {
            "full_match_rows_using_track_counts": [{"lps_track_id": track, "full_tcc_rows": n,
                "percent_of_full_matches": round(100 * n / full_count, 3) if full_count else 0.0}
                for track, n in track_rows[:20]],
            "largest_track_full_row_share_percent": round(100 * largest_track_rows / full_count, 3) if full_count else 0.0,
            "warning_threshold_percent": 10.0,
            "exceeds_warning_threshold": bool(full_count and 100 * largest_track_rows / full_count > 10),
            "note": "Overlapping TCC issue windows naturally reuse track positions; concentration is a diagnostic, not independent-sample count.",
        },
        "predictor_availability_on_full_matches": {**feature_coverage,
            "tcc_rows_complete_9_of_9_for_both": int(both_complete)},
        "semester_project_screen": {
            "criteria": "at least 100 full rows, 20 unique TCC tracks, 20 rows in each label, at least 10 of each label in every frozen split, both predictors complete on at least 90% of full rows, and no LPS track used by more than 10% of full rows",
            "label_counts": label_counts, "full_rows_by_split_and_label": class_split,
            "both_predictors_complete_rate": round(both_complete_rate, 4),
            "adequate_by_screen": bool(semester_ready),
            "note": "Heuristic screening only; whole-TCC-track splits remain fixed and no random split is created.",
        },
        "verdict": verdict,
        "representative_full_matches": rows.loc[rows.full_9_step_match].head(5).to_dict(orient="records"),
        "representative_unmatched": rows.loc[rows.matched_timestamps.eq(0)].head(5).to_dict(orient="records"),
        "representative_partial_matches": rows.loc[rows.matched_timestamps.between(1, 8)].head(5).to_dict(orient="records"),
        "lps_intensity_or_category_used_as_target_or_predictor": False,
        "tcc_labels_and_splits_modified": False,
    }
    return rows, positions, summary


def write_report(path: Path, summary: dict) -> None:
    if summary.get("status") != "complete":
        path.write_text(
            "# LPS–TCC crosswalk status\n\n"
            f"**Status:** {summary['status']}\n\n"
            f"{summary['reason']}\n\n"
            "No match counts are available; no cohort, labels, or splits were changed.\n"
        )
        return
    rows = summary["tcc_rows_examined"]
    full = summary["full_9_of_9_matches"]
    lines = [
        "# South Asian LPS–TCC metadata crosswalk",
        "",
        f"- TCC rows examined: {rows}; zero/partial/full matches: {summary['zero_matches']}/{summary['partial_1_to_8_matches']}/{full}.",
        f"- Full 9/9 coverage: {summary['full_match_percent']:.3f}%.",
        f"- Accepted unique-position distances (km): {summary['accepted_unique_position_distance_km']}.",
        f"- Ambiguous TCC positions: {summary['ambiguous_unique_tcc_positions']}; non-detected matches used: 0.",
        f"- Unique LPS tracks used: {summary['unique_lps_track_ids_used']}; full rows with both features complete: {summary['predictor_availability_on_full_matches']['tcc_rows_complete_9_of_9_for_both']}.",
        "",
        "## Full-match rates",
        "",
        "| Grouping | Group | Rows | Full 9/9 | Percent |",
        "|---|---:|---:|---:|---:|",
    ]
    for title, key in (("Basin", "full_matches_by_basin"), ("Split", "full_matches_by_split"), ("24h label", "full_matches_by_label_24h")):
        for item in summary[key]:
            lines.append(f"| {title} | {item['group']} | {item['rows']} | {item['full_9_step_rows']} | {item['full_9_step_percent']:.3f}% |")
    lines += ["", "## Retained subset", "",
              f"Full-match rows: {summary['retained_subset']['full_9_of_9_rows']}; unique TCC tracks: {summary['retained_subset']['unique_tcc_tracks']}; years: {summary['retained_subset']['years']}.",
              f"Frozen split/label counts: {summary['retained_subset']['split_label_counts']}.",
              f"Semester-project screening rule: {summary['semester_project_screen']['criteria']}.",
              "LPS intensity/category fields were not used as predictors or labels. TCC labels and splits are unchanged.", "",
              f"**{ {'A': 'A — viable enough to proceed with a regional matched-TCC experiment', 'B': 'B — technically matchable but coverage is too weak/biased for a useful experiment', 'C': 'C — crosswalk fails; abandon this catalogue for TCC enrichment'}[summary['verdict']] }**", ""]
    path.write_text("\n".join(lines))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/genesis_candidate_manifest.csv"))
    parser.add_argument("--tcc-source", type=Path, required=True, help="merged TCC/IBTrACS CSV containing source=1 rows")
    parser.add_argument("--lps-catalogue", type=Path, required=True, help="local LPS Parquet or CSV/CSV.GZ; not downloaded by this tool")
    parser.add_argument("--out-dir", type=Path, default=Path("results/lps_tcc_crosswalk"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "audit.json"
    report_path = args.out_dir / "report.md"
    if not args.lps_catalogue.exists():
        summary = {"status": "blocked_input_unavailable", "tcc_rows_examined": 1276,
                   "lps_catalogue_path": str(args.lps_catalogue),
                   "reason": "No local LPS catalogue payload is available; only the published Zenodo schema metadata was inspected. Command-line DNS/network access to Zenodo failed, and the 291.6 MB payload was not downloaded.",
                   "required_lps_columns": list(LPS_COLUMNS), "crosswalk_metrics": None,
                   "catalogue_bytes_downloaded": 0,
                   "tcc_labels_and_splits_modified": False}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        write_report(report_path, summary)
        print(json.dumps(summary, indent=2))
        raise SystemExit(2)

    manifest = cohort(_read_table(args.manifest))
    tcc = load_tcc(args.tcc_source)
    lps = read_lps_catalogue(args.lps_catalogue)
    rows, positions, summary = build_crosswalk(manifest, tcc, lps)
    rows.to_csv(args.out_dir / "tcc_rows.csv", index=False)
    positions.to_csv(args.out_dir / "tcc_positions.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_report(report_path, summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
