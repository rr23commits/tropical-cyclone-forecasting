"""Focused checks for strict LPS-to-TCC timestamp/position matching."""

import unittest

import pandas as pd

from src.audit_lps_tcc_crosswalk import _detected_rows, build_crosswalk


def fixture(track_ids=(10,), duplicate_at=None):
    times = pd.date_range("2001-08-01T00:00:00Z", periods=9, freq="3h")
    manifest, tcc, lps = [], [], []
    lps_id = 0
    for track in track_ids:
        lat_offset = 0.0 if track == 10 else 0.01
        for step, stamp in enumerate(times):
            lat, lon = 15.0 + step * 0.1 + lat_offset, 80.0 + step * 0.1
            tcc.append({"number": track, "source": 1, "time": stamp, "lat": lat, "lon": lon, "devflag": 1})
            lps.append({"row_id": lps_id, "track_id": 700, "time": stamp, "lat_detected": lat + 0.001,
                        "lon_detected": lon, "position_source": "observed", "mean_vort_850_detectedcentre": 1.5,
                        "rh700_mean_pct_detectedcentre": 65.0})
            lps_id += 1
            if duplicate_at == step:
                lps.append({"row_id": lps_id, "track_id": 701, "time": stamp, "lat_detected": lat - 0.001,
                            "lon_detected": lon, "position_source": "observed", "mean_vort_850_detectedcentre": 1.0,
                            "rh700_mean_pct_detectedcentre": 60.0})
                lps_id += 1
        manifest.append({"tcc_track_id": track, "basin": "NI", "year": 2001,
                         "issue_time_utc": times[-1].isoformat().replace("+00:00", "Z"),
                         "current_lat": 15.8 + lat_offset, "current_lon": 80.8,
                         "linked_ibtracs_sid": "", "first_34kt_time_utc": "", "label_24h": 1,
                         "label_48h": 1, "input_times_utc": "|".join(x.isoformat().replace("+00:00", "Z") for x in times),
                         "split": "train"})
    return pd.DataFrame(manifest), pd.DataFrame(tcc), pd.DataFrame(lps)


class LpsTccCrosswalkTests(unittest.TestCase):
    def test_row_id_may_repeat_at_different_times_but_not_same_time(self):
        rows = pd.DataFrame({
            "row_id": [7, 7], "time": pd.to_datetime(["2001-08-01T00:00Z", "2001-08-01T03:00Z"]),
            "lat_detected": [15.0, 15.1], "lon_detected": [80.0, 80.1],
            "position_source": ["observed", "observed"],
        })
        detected, _ = _detected_rows(rows)
        self.assertEqual(len(detected), 2)
        rows.loc[1, "time"] = rows.loc[0, "time"]
        with self.assertRaisesRegex(ValueError, "unique, non-null \\(time, row_id\\)"):
            _detected_rows(rows)

    def test_full_match_uses_detected_positions_and_tracks_feature_nulls(self):
        manifest, tcc, lps = fixture()
        lps.loc[lps.time.eq(pd.Timestamp("2001-08-01T06:00:00Z")), "mean_vort_850_detectedcentre"] = None
        rows, positions, summary = build_crosswalk(manifest, tcc, lps)
        self.assertEqual(rows.loc[0, "matched_timestamps"], 9)
        self.assertTrue(rows.loc[0, "full_9_step_match"])
        self.assertEqual(summary["full_9_of_9_matches"], 1)
        self.assertEqual(summary["predictor_availability_on_full_matches"]["mean_vort_850_detectedcentre"]["rows_with_9_of_9"], 0)
        self.assertEqual(summary["predictor_availability_on_full_matches"]["tcc_rows_complete_9_of_9_for_both"], 0)
        self.assertTrue(positions.matched.all())
        self.assertEqual(summary["unique_lps_track_ids_used"], 1)

    def test_multiple_nearby_detections_are_ambiguous_not_selected(self):
        manifest, tcc, lps = fixture(duplicate_at=4)
        rows, positions, summary = build_crosswalk(manifest, tcc, lps)
        self.assertEqual(rows.loc[0, "matched_timestamps"], 8)
        ambiguous = positions.loc[positions.ambiguous]
        self.assertEqual(len(ambiguous), 1)
        self.assertEqual(ambiguous.iloc[0].candidate_count_within_1deg, 2)
        self.assertEqual(summary["ambiguous_unique_tcc_positions"], 1)

    def test_one_lps_detection_cannot_be_assigned_to_two_tcc_positions(self):
        manifest, tcc, lps = fixture(track_ids=(10, 11))
        lps = lps.loc[lps.row_id.lt(9)].copy()
        rows, positions, summary = build_crosswalk(manifest, tcc, lps)
        self.assertTrue(rows.matched_timestamps.eq(0).all())
        self.assertTrue(positions.match_status.eq("ambiguous_shared_lps_detection").all())
        self.assertEqual(summary["unique_lps_track_ids_used"], 0)

    def test_posterior_rows_are_never_used(self):
        manifest, tcc, lps = fixture()
        lps.loc[0, ["lat_detected", "lon_detected"]] = None
        lps.loc[0, "position_source"] = "posterior_interpolated"
        rows, positions, summary = build_crosswalk(manifest, tcc, lps)
        self.assertEqual(rows.loc[0, "matched_timestamps"], 8)
        self.assertEqual(summary["matches_using_non_detected_or_interpolated_rows"], 0)
        self.assertEqual(summary["non_detected_lps_rows_at_required_timestamps_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
