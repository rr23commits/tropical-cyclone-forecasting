"""Focused checks for held-out error-analysis labels and rows."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.error_analysis import error_rows, intensity_bin, motion_bin, summarize, wind_change_bin
from src.ibtracs import ForecastSamples


class ErrorAnalysisTests(unittest.TestCase):
    def test_descriptive_bins_cover_boundaries(self) -> None:
        self.assertEqual(intensity_bin(pd.Series([33, 34, 113])).tolist(), ["TD (<34)", "TS (34-63)", "Major (>=113)"])
        self.assertEqual(motion_bin(pd.Series([10, 25, 26])).tolist(), ["slow (<10)", "typical (10-25)", "fast (>=25)"])
        self.assertEqual(wind_change_bin(pd.Series([-6, 0, 6])).tolist(), ["weakening (<-5)", "steady (-5 to 5)", "strengthening (>5)"])

    def test_perfect_direct_prediction_produces_zero_error_rows(self) -> None:
        metadata = pd.DataFrame({
            "SID": ["A"], "SEASON": [2020],
            "issue_time": pd.to_datetime(["2020-08-01T00:00:00Z"]), "target_time": pd.to_datetime(["2020-08-01T06:00:00Z"]),
            "issue_lat": [10.0], "issue_lon": [-50.0], "issue_wind": [40.0],
            "target_lat": [10.0], "target_lon": [-50.0], "target_wind": [40.0],
        })
        samples = ForecastSamples(np.zeros((1, 5, 9)), np.zeros((1, 3)), metadata, np.empty((1, 5), dtype=object), 6)
        rows = error_rows(samples, np.zeros((1, 3)), "test")
        self.assertEqual(rows.loc[0, "track_error_km"], 0.0)
        self.assertEqual(rows.loc[0, "north_error_km"], 0.0)
        self.assertEqual(rows.loc[0, "east_error_km"], 0.0)
        self.assertEqual(rows.loc[0, "wind_abs_error_kt"], 0.0)

    def test_summary_uses_storm_level_interval(self) -> None:
        rows = pd.DataFrame({
            "model": ["test", "test", "test"], "horizon_hours": [6, 6, 6], "SID": ["A", "A", "B"],
            "track_error_km": [10.0, 30.0, 50.0], "wind_abs_error_kt": [1.0, 3.0, 5.0], "wind_error_kt": [1.0, 3.0, 5.0],
        })
        summary = summarize(rows, ["model", "horizon_hours"], bootstrap_samples=10)
        self.assertEqual(summary.loc[0, "storm_macro_track_mean_km"], 35.0)
        self.assertLessEqual(summary.loc[0, "storm_track_ci95_low_km"], 35.0)
        self.assertGreaterEqual(summary.loc[0, "storm_track_ci95_high_km"], 35.0)


if __name__ == "__main__":
    unittest.main()
