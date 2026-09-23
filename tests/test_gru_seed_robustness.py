import unittest

import pandas as pd

from src.run_gru_seed_robustness import summarize_seed_results


class GRUSeedRobustnessTests(unittest.TestCase):
    def test_summary_equally_aggregates_all_fixed_seed_results(self):
        rows = pd.DataFrame([
            {"horizon_hours": 6, "seed": 42, "origins": 2, "storms": 1, "track_error_km": 10.0, "wind_mae": 2.0, "wind_rmse": 3.0},
            {"horizon_hours": 6, "seed": 43, "origins": 2, "storms": 1, "track_error_km": 16.0, "wind_mae": 4.0, "wind_rmse": 5.0},
            {"horizon_hours": 6, "seed": 44, "origins": 2, "storms": 1, "track_error_km": 13.0, "wind_mae": 3.0, "wind_rmse": 4.0},
        ])
        summary = summarize_seed_results(rows).iloc[0]
        self.assertEqual(summary.seeds, 3)
        self.assertEqual(summary.track_error_km_mean, 13.0)
        self.assertEqual(summary.wind_mae_mean, 3.0)
        self.assertGreater(summary.track_error_km_std, 0.0)
