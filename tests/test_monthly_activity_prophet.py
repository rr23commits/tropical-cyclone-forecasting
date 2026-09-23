import unittest

import numpy as np
import pandas as pd

from src.monthly_activity_prophet import hyperparameter_grid, prophet_history


class ProphetFeatureTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("1980-01-01", "2025-12-01", freq="MS", tz="UTC")
        self.series = pd.Series(np.arange(len(index), dtype=float), index=index)
        self.target = pd.Timestamp("2020-01-01", tz="UTC")

    def test_prophet_history_uses_only_past_months(self):
        changed = self.series.copy()
        changed.loc[self.target:] = 9999
        left = prophet_history(self.series, self.target)
        right = prophet_history(changed, self.target)
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(left.ds.iloc[-1], pd.Timestamp("2019-12-01"))

    def test_history_has_monthly_ds_and_count_y(self):
        history = prophet_history(self.series, self.target)
        self.assertEqual(list(history.columns), ["ds", "y"])
        self.assertEqual(history.ds.dt.day.unique().tolist(), [1])

    def test_grid_is_minimal_and_has_no_regressors(self):
        grid = hyperparameter_grid()
        self.assertEqual(len(grid), 4)
        self.assertEqual({item["seasonality_mode"] for item in grid}, {"additive"})
        self.assertEqual({item["yearly_fourier_order"] for item in grid}, {5})


if __name__ == "__main__":
    unittest.main()
