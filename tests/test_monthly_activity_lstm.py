import unittest

import numpy as np
import pandas as pd

from src.monthly_activity_lstm import SEQUENCE_LENGTH, hyperparameter_grid, sequence_for_target, training_sequences


class LSTMSequenceTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("1980-01-01", "2025-12-01", freq="MS", tz="UTC")
        self.series = pd.Series(np.arange(len(index), dtype=float), index=index)
        self.target = pd.Timestamp("2020-01-01", tz="UTC")

    def test_sequence_is_exactly_12_past_values(self):
        sequence = sequence_for_target(self.series, self.target)
        self.assertEqual(len(sequence), SEQUENCE_LENGTH)
        self.assertTrue(np.array_equal(sequence, self.series.loc["2019-01-01":"2019-12-01"].to_numpy(dtype=np.float32)))

    def test_sequence_ignores_target_and_future_values(self):
        changed = self.series.copy()
        changed.loc[self.target:] = 9999
        self.assertTrue(np.array_equal(sequence_for_target(self.series, self.target), sequence_for_target(changed, self.target)))

    def test_training_targets_stop_before_forecast_month(self):
        x, y = training_sequences(self.series, self.target, 0.0, 1.0)
        self.assertEqual(len(x), len(y))
        self.assertEqual(y[-1, 0], self.series.loc["2019-12-01"])

    def test_small_grid_tunes_only_small_architecture_settings(self):
        grid = hyperparameter_grid()
        self.assertEqual(len(grid), 4)
        self.assertEqual({item["hidden_size"] for item in grid}, {4, 8})
        self.assertEqual({item["learning_rate"] for item in grid}, {0.005, 0.01})


if __name__ == "__main__":
    unittest.main()
