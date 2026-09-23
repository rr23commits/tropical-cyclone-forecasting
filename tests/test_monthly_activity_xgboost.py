import unittest

import numpy as np
import pandas as pd

from src.monthly_activity_xgboost import feature_row, feature_frame, hyperparameter_grid


class XGBoostFeatureTests(unittest.TestCase):
    def test_features_use_only_past_values(self):
        index = pd.date_range("1980-01-01", "2025-12-01", freq="MS", tz="UTC")
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        target = pd.Timestamp("2020-01-01", tz="UTC")
        changed = series.copy()
        changed.loc[target:] = 9999
        self.assertEqual(feature_row(series, target, True), feature_row(changed, target, True))

    def test_trailing_mean_excludes_target_month(self):
        index = pd.date_range("1980-01-01", "2025-12-01", freq="MS", tz="UTC")
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        target = pd.Timestamp("2020-01-01", tz="UTC")
        row = feature_row(series, target, True)
        self.assertEqual(row["trailing_mean_3"], float(series.loc["2019-10-01":"2019-12-01"].mean()))
        self.assertNotIn("trailing_mean_3", feature_row(series, target, False))

    def test_features_have_requested_lags_and_month_encoding(self):
        index = pd.date_range("1980-01-01", "2025-12-01", freq="MS", tz="UTC")
        series = pd.Series(np.ones(len(index)), index=index)
        frame = feature_frame(series, pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")]), True)
        self.assertEqual(set(frame.columns), {"lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "month_sin", "month_cos", "trailing_mean_3"})

    def test_grid_tunes_trailing_mean_and_tree_settings(self):
        grid = hyperparameter_grid()
        self.assertGreater(len(grid), 1)
        self.assertEqual({item["trailing_mean"] for item in grid}, {False, True})
        self.assertGreater(len({item["max_depth"] for item in grid}), 1)


if __name__ == "__main__":
    unittest.main()
