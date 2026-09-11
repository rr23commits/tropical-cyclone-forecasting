"""Focused Phase 2 baseline and evaluation tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.baselines import constant_motion, fit_ridge, flattened_features, persistence
from src.evaluation import align_common_origins, evaluate, split_samples
from src.ibtracs import ForecastSamples


def samples() -> ForecastSamples:
    features = np.zeros((2, 5, 9), dtype=float)
    features[:, -1, 3:5] = [10.0, -5.0]
    metadata = pd.DataFrame({
        "SID": ["TRAIN", "TEST"],
        "issue_time": pd.to_datetime(["2000-01-01T00:00:00Z", "2000-01-01T06:00:00Z"]),
        "target_time": pd.to_datetime(["2000-01-02T00:00:00Z", "2000-01-02T06:00:00Z"]),
        "issue_lat": [10.0, 10.0], "issue_lon": [-50.0, -50.0], "issue_wind": [40.0, 40.0],
        "target_lat": [10.0, 10.0], "target_lon": [-50.0, -50.0], "target_wind": [40.0, 40.0],
    })
    return ForecastSamples(features, np.zeros((2, 3)), metadata, np.empty((2, 5), dtype=object), 24)


class BaselineTests(unittest.TestCase):
    def test_persistence_is_zero_change(self) -> None:
        self.assertTrue(np.array_equal(persistence(samples()), np.zeros((2, 3))))

    def test_constant_motion_scales_last_six_hour_displacement(self) -> None:
        prediction = constant_motion(samples())
        np.testing.assert_allclose(prediction, [[40.0, -20.0, 0.0], [40.0, -20.0, 0.0]])

    def test_perfect_prediction_has_zero_evaluation_errors(self) -> None:
        data = samples()
        metrics = evaluate(data, np.zeros((2, 3)))
        self.assertAlmostEqual(metrics["track_error_km"], 0.0)
        self.assertAlmostEqual(metrics["wind_mae"], 0.0)
        self.assertAlmostEqual(metrics["wind_rmse"], 0.0)

    def test_split_samples_keeps_storms_atomic(self) -> None:
        groups = split_samples(samples(), pd.Series({"TRAIN": "train", "TEST": "test"}))
        self.assertEqual(groups["train"].metadata.SID.tolist(), ["TRAIN"])
        self.assertEqual(groups["validation"].metadata.shape[0], 0)
        self.assertEqual(groups["test"].metadata.SID.tolist(), ["TEST"])

    def test_alignment_keeps_only_common_storm_origins(self) -> None:
        first = samples()
        second = ForecastSamples(
            first.features[1:], first.targets[1:], first.metadata.iloc[1:].reset_index(drop=True),
            first.history_times[1:], first.horizon_hours,
        )
        aligned_first, aligned_second = align_common_origins([first, second])
        self.assertEqual(aligned_first.metadata.SID.tolist(), ["TEST"])
        self.assertEqual(aligned_second.metadata.SID.tolist(), ["TEST"])

    def test_ridge_fits_direct_displacement_and_wind_change(self) -> None:
        data = samples()
        model = fit_ridge(flattened_features(data), np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]), alpha=1.0)
        prediction = model.predict(flattened_features(data))
        self.assertEqual(prediction.shape, (2, 3))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
