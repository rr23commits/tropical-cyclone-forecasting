"""Focused checks for metadata-only genesis labels and temporal splits."""

from __future__ import annotations

import unittest

import pandas as pd

from src.audit_genesis import _links, build_manifest


class GenesisAuditTests(unittest.TestCase):
    def test_labels_history_and_boundary_purge(self) -> None:
        rows = []
        for number, start, end in (
            (1, "2007-12-30 00:00", "2008-01-01 00:00"),
            (2, "2008-12-31 00:00", "2009-01-05 00:00"),
            (3, "2009-01-01 00:00", "2009-01-06 00:00"),
        ):
            times = pd.date_range(start, end, freq="3h", tz="UTC")
            lat = 30.0 if number == 3 else 10.0
            rows.extend({"number": number, "source": 1, "time": time, "lat": lat, "lon": -50.0,
                         "devflag": int(number == 2)}
                        for time in times)
        rows.extend({"number": 2, "source": 2, "time": time, "lat": 10.0, "lon": -50.0,
                     "devflag": 1} for time in pd.date_range("2009-01-05 03:00", "2009-01-08", freq="3h", tz="UTC"))
        tcc = pd.DataFrame(rows)
        ib = pd.DataFrame([{
            "SID": "SID1", "BASIN": "NA", "ISO_TIME": time, "TRACK_TYPE": "main",
            "LAT": 10.0, "LON": -50.0, "USA_WIND": 20 if time < pd.Timestamp("2009-01-02T00:00Z") else 35,
        } for time in pd.date_range("2008-12-31", "2009-01-08", freq="6h", tz="UTC")])

        manifest, counts = build_manifest(tcc, ib)
        later = manifest.loc[manifest.tcc_track_id.eq(2)].set_index("issue_time_utc")
        positive = later.loc["2009-01-01T00:00:00Z"]
        negative = manifest.loc[(manifest.tcc_track_id == 3) &
                                (manifest.issue_time_utc == "2009-01-02T00:00:00Z")].iloc[0]

        self.assertEqual(positive.label_24h, 1)
        self.assertEqual(positive.label_48h, 1)
        self.assertEqual(positive.linked_ibtracs_sid, "SID1")
        input_times = pd.to_datetime(positive.input_times_utc.split("|"), utc=True)
        self.assertEqual(len(input_times), 9)
        self.assertEqual(input_times[-1], pd.Timestamp("2009-01-01T00:00:00Z"))
        self.assertTrue((input_times <= pd.Timestamp("2009-01-01T00:00:00Z")).all())
        self.assertEqual(negative.label_24h, 0)
        self.assertEqual(negative.label_48h, 0)
        self.assertEqual(negative.linked_ibtracs_sid, "")
        self.assertNotIn("2008-01-01T00:00:00Z", manifest.issue_time_utc.tolist())
        self.assertGreater(counts["split_range_or_boundary_purged"], 0)
        self.assertTrue((manifest.split == "validation").all())

    def test_linked_negative_needs_wind_coverage_through_target(self) -> None:
        tcc = pd.DataFrame({
            "number": 9, "source": 1, "devflag": 1,
            "time": pd.date_range("2008-12-31", "2009-01-04", freq="3h", tz="UTC"),
            "lat": 10.0, "lon": -50.0,
        })
        bridge = pd.DataFrame([{
            "number": 9, "source": 2, "time": time, "lat": 10.0, "lon": -50.0, "devflag": 1,
        } for time in pd.to_datetime(["2009-01-04T03:00Z", "2009-01-04T09:00Z"])])
        tcc = pd.concat([tcc, bridge], ignore_index=True)
        ib = pd.DataFrame([{
            "SID": "SID9", "BASIN": "NA", "ISO_TIME": time, "TRACK_TYPE": "main",
            "LAT": 10.0, "LON": -50.0, "USA_WIND": 20,
        } for time in pd.to_datetime(["2009-01-04T03:00Z", "2009-01-04T09:00Z"])])

        manifest, counts = build_manifest(tcc, ib)

        self.assertTrue(manifest.empty)
        self.assertGreater(counts["insufficient_24h_followup"], 0)

    def test_single_point_sid_match_is_not_used(self) -> None:
        bridge = pd.DataFrame([{
            "number": 8, "source": 2, "time": pd.Timestamp("2009-01-01T00:00Z"),
            "lat": 10.0, "lon": -50.0,
        }])
        ib = pd.DataFrame([{
            "SID": "SID8", "BASIN": "NA", "ISO_TIME": pd.Timestamp("2009-01-01T00:00Z"),
            "TRACK_TYPE": "main", "LAT": 10.0, "LON": -50.0, "USA_WIND": 20,
        }])

        links, diagnostics = _links(bridge, ib)

        self.assertEqual(links, {})
        self.assertEqual(diagnostics["single_point_links_excluded"], 1)

    def test_exact_sid_link_tie_fails_closed(self) -> None:
        bridge = pd.DataFrame([{
            "number": 7, "source": 2, "time": pd.Timestamp("2009-01-01T00:00Z"),
            "lat": 10.0, "lon": -50.0,
        }])
        ib = pd.DataFrame([{
            "SID": sid, "BASIN": "NA", "ISO_TIME": pd.Timestamp("2009-01-01T00:00Z"),
            "TRACK_TYPE": "main", "LAT": 10.0, "LON": -50.0, "USA_WIND": 20,
        } for sid in ("SID_A", "SID_B")])

        links, diagnostics = _links(bridge, ib)

        self.assertEqual(links, {})
        self.assertEqual(diagnostics["unresolved_exact_ties"], 1)


if __name__ == "__main__":
    unittest.main()
