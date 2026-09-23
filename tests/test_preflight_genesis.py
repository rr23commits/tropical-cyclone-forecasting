from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.preflight_genesis import build_report, cached_gridsat_catalogs, cohort, required_times


class GenesisPreflightTests(unittest.TestCase):
    def test_cached_annual_index_is_reused_without_network(self) -> None:
        stamp = pd.Timestamp("2000-01-01T00:00Z")
        with TemporaryDirectory() as directory, patch(
            "src.preflight_genesis.fetch_gridsat_catalogs", return_value={stamp: 12}
        ) as fetch:
            first = cached_gridsat_catalogs(pd.DatetimeIndex([stamp]), Path(directory))
            second = cached_gridsat_catalogs(pd.DatetimeIndex([stamp]), Path(directory))
        self.assertEqual(first, {stamp: 12})
        self.assertEqual(second, {stamp: 12})
        fetch.assert_called_once()

    def test_missing_catalog_frame_blocks_only_affected_row(self) -> None:
        times = pd.date_range("2000-01-01", periods=10, freq="3h", tz="UTC")
        rows = pd.DataFrame({"basin": ["NA"] * 1276, "current_lat": [10.0] * 1276,
                             "current_lon": [0.0] * 1276, "tcc_track_id": range(1276),
                             "issue_time_utc": ["2000-01-02T03:00:00Z"] * 1276,
                             "input_times_utc": ["|".join(map(lambda t: t.isoformat().replace("+00:00", "Z"), times[:9]))] * 1275 +
                                                ["|".join(map(lambda t: t.isoformat().replace("+00:00", "Z"), times[1:]))]})
        available = {stamp: 10 for stamp in required_times(cohort(rows))[1:]}
        report = build_report(rows, available)
        self.assertEqual(report["gridsat"]["missing_frames"], 1)
        self.assertEqual(report["gridsat"]["complete_history_rows"], 1)
        self.assertEqual(report["decision"], "NO-GO")
        self.assertEqual(len(report["row_blockers"]), 1275)


if __name__ == "__main__":
    unittest.main()
