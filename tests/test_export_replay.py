import unittest

import pandas as pd

from src.export_replay import build_payload


class ReplayExportTests(unittest.TestCase):
    def test_groups_observed_states_and_model_predictions(self):
        rows = pd.DataFrame([
            {"SID": "S", "SEASON": 2020, "issue_time": "2020-01-01", "target_time": "2020-01-01 06:00", "issue_lat": 10, "issue_lon": -50, "issue_wind": 40, "target_lat": 11, "target_lon": -49, "target_wind": 45, "model": model, "horizon_hours": 6, "predicted_lat": 10.5, "predicted_lon": -49.5, "predicted_wind": 43, "track_error_km": 10, "wind_abs_error_kt": 2}
            for model in ("persistence", "constant-motion", "ridge", "gru")
        ])
        metrics = pd.DataFrame([{ "model": "gru", "horizon_hours": 6, "origins": 1, "storms": 1, "track_mean_km": 10, "wind_mae_kt": 2 }])
        payload = build_payload(rows, metrics, {"S": {"name": "ALPHA", "basin": "NA", "stormId": "AL012020", "season": 2020}})
        self.assertEqual(len(payload["storms"][0]["observed"]), 2)
        self.assertEqual(set(payload["storms"][0]["forecasts"][0]["predictions"]), {"persistence", "constant-motion", "ridge", "gru"})
        self.assertEqual(payload["storms"][0]["stormId"], "AL012020")
