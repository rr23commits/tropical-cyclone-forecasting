import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from src.monthly_activity import END, START, TRAIN_END, VALIDATION_END, _metrics
from src.monthly_activity_autots import count_forecast, expanding_forecast, history_frame, search_templates, select_template
from src.export_monthly_frontend import build_payload


class AutoTSMonthlyTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range(START, END, freq="MS", tz="UTC")
        self.series = pd.Series(np.arange(len(index), dtype=float) % 5, index=index)
        self.templates = pd.DataFrame({"Model": ["A", "B"], "ModelParameters": ["{}", "{}"]})

    def test_expanding_output_has_shared_dates_and_nonnegative_whole_counts(self):
        months = pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-02-01", tz="UTC")])
        with patch("src.monthly_activity_autots._fixed_template_predict", side_effect=[-2.0, 2.5]):
            forecast = expanding_forecast(self.series, months, self.templates.iloc[[0]])
        self.assertListEqual(list(forecast.index), list(months))
        self.assertListEqual(forecast.tolist(), [0.0, 3.0])

    def test_missing_or_failed_prediction_uses_nonnegative_seasonal_fallback(self):
        month = pd.Timestamp("2020-01-01", tz="UTC")
        failures = []
        with patch("src.monthly_activity_autots._fixed_template_predict", side_effect=RuntimeError("candidate failed")):
            forecast = expanding_forecast(self.series, pd.DatetimeIndex([month]), self.templates.iloc[[0]], failures)
        self.assertEqual(len(failures), 1)
        self.assertGreaterEqual(forecast.iloc[0], 0)
        self.assertEqual(count_forecast(float("nan"), -1), 0.0)

    def test_history_excludes_current_and_future_observations(self):
        target = pd.Timestamp("2020-01-01", tz="UTC")
        changed = self.series.copy()
        changed.loc[target:] = 9999
        pd.testing.assert_frame_equal(history_frame(self.series, target), history_frame(changed, target))
        self.assertEqual(history_frame(self.series, target).index[-1], pd.Timestamp("2019-12-01"))

    def test_validation_selection_never_requests_test_months(self):
        seen = []

        def fake_forecast(series, months, template, failures=None):
            seen.extend(months)
            return pd.Series(np.zeros(len(months)), index=months)

        with patch("src.monthly_activity_autots.expanding_forecast", side_effect=fake_forecast):
            select_template(self.series, self.templates)
        self.assertTrue(seen)
        self.assertTrue(all(TRAIN_END < month <= VALIDATION_END for month in seen))

    def test_metrics_are_shared_monthly_mae_rmse_and_mase(self):
        actual = pd.Series([1.0, 3.0])
        prediction = pd.Series([0.0, 5.0])
        metrics = _metrics(actual, prediction, pd.Series([0.0, 2.0, 4.0]))
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5))
        self.assertEqual(metrics["mase"], 0.75)

    def test_frontend_payload_includes_autots_only_when_test_artifact_exists(self):
        with TemporaryDirectory() as directory:
            # Keep this fixture deliberately small: the exporter only requires
            # the same CSV schema written by the experiment.
            root = Path(directory)
            pd.DataFrame({"month": ["2020-01-01"], "storm_count": [1]}).to_csv(root / "monthly_series.csv", index=False)
            pd.DataFrame({"month": ["2020-01-01"], "actual": [1], "trend": [1], "seasonal": [0], "residual": [0]}).to_csv(root / "decomposition.csv", index=False)
            pd.DataFrame({"model": ["autots"], "split": ["test"], "n": [1], "mae": [0], "rmse": [0], "mase": [0], "wape": [float("nan")]}).to_csv(root / "metrics.csv", index=False)
            pd.DataFrame({"month": ["2020-01-01"], "actual": [1], "forecast": [1], "model": ["autots"], "split": ["test"]}).to_csv(root / "autots_forecasts.csv", index=False)
            pd.DataFrame({"model": ["autots"], "split": ["test"], "n": [1], "mae": [0], "rmse": [0], "mase": [0]}).to_csv(root / "autots_metrics.csv", index=False)
            # Ensemble remains a required existing artifact.
            pd.DataFrame({"month": ["2020-01-01"], "actual": [1], "forecast": [1], "model": ["ensemble_xgboost_prophet"], "split": ["test"]}).to_csv(root / "ensemble_forecasts.csv", index=False)
            pd.DataFrame({"model": ["ensemble_xgboost_prophet"], "split": ["test"], "n": [1], "mae": [0], "rmse": [0], "mase": [0]}).to_csv(root / "ensemble_metrics.csv", index=False)
            payload = build_payload(root)
        self.assertIn("autots", payload["models"])
        self.assertEqual(payload["forecasts"]["autots"][0]["model"], "autots")
        self.assertIsNone(payload["metrics"][0]["wape"])

    def test_search_exports_to_autots_string_path_api(self):
        class FakeAutoTS:
            exported = None
            def __init__(self, **kwargs): pass
            def fit(self, frame): return self
            def export_template(self, path, **kwargs):
                self.exported = path
                pd.DataFrame({"Model": ["LastValueNaive"]}).to_csv(path, index=False)

        with TemporaryDirectory() as directory, patch("src.monthly_activity_autots._autots_class", return_value=FakeAutoTS):
            templates = search_templates(self.series.loc[:TRAIN_END], Path(directory))
            self.assertEqual(len(templates), 1)


if __name__ == "__main__":
    unittest.main()
