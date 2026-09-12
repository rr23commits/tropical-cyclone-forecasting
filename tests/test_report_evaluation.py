import unittest

import pandas as pd

from src.report_evaluation import paired_gru_ridge


class EvaluationReportTests(unittest.TestCase):
    def test_paired_comparison_uses_shared_origins_and_storms(self):
        rows = pd.DataFrame([
            {"SID": sid, "issue_time": time, "target_time": time, "horizon_hours": 6, "model": model,
             "track_error_km": track, "wind_abs_error_kt": wind}
            for sid, time, model, track, wind in [
                ("A", "a", "gru", 8, 2), ("A", "a", "ridge", 10, 3),
                ("B", "b", "gru", 14, 5), ("B", "b", "ridge", 12, 4),
            ]
        ])
        result = paired_gru_ridge(rows, bootstrap_samples=20).iloc[0]
        self.assertEqual(result.common_origins, 2)
        self.assertEqual(result.common_storms, 2)
        self.assertEqual(result.track_delta_gru_minus_ridge_km, 0.0)
        self.assertEqual(result.wind_mae_delta_gru_minus_ridge_kt, 0.0)
