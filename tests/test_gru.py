"""Focused tests for compact GRU scaling, output shape, and held-out prediction."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.gru import GRUForecast, GRUSettings, SequenceScaler, predict_gru, train_select_gru
from src.ibtracs import ForecastSamples


def make_samples(count: int, offset: float = 0.0) -> ForecastSamples:
    features = np.arange(count * 5 * 9, dtype=float).reshape(count, 5, 9) / 10.0 + offset
    targets = np.column_stack((features[:, -1, 0], features[:, -1, 1], features[:, -1, 2]))
    metadata = pd.DataFrame({
        "SID": [f"S{index}" for index in range(count)],
        "issue_lat": np.zeros(count), "issue_lon": np.zeros(count), "issue_wind": np.zeros(count),
        "target_lat": np.zeros(count), "target_lon": np.zeros(count), "target_wind": np.zeros(count),
    })
    return ForecastSamples(features, targets, metadata, np.empty((count, 5), dtype=object), 6)


class GRUTests(unittest.TestCase):
    def test_scaler_is_fit_from_training_samples_only(self) -> None:
        train = make_samples(3)
        validation = make_samples(2, offset=1000.0)
        scaler = SequenceScaler.fit(train)
        np.testing.assert_allclose(scaler.transform_features(train.features).mean(axis=(0, 1)), 0.0, atol=1e-6)
        self.assertGreater(scaler.transform_features(validation.features).mean(), 100.0)

    def test_gru_returns_three_direct_outputs(self) -> None:
        model = GRUForecast(input_size=9, hidden_size=16)
        self.assertEqual(tuple(model(torch.zeros((2, 5, 9))).shape), (2, 3))

    def test_training_and_prediction_do_not_require_test_targets(self) -> None:
        train, validation, test = make_samples(6), make_samples(3, 1.0), make_samples(2, 2.0)
        settings = GRUSettings(hidden_sizes=(4,), max_epochs=3, patience=2, batch_size=3)
        trained = train_select_gru(train, validation, settings)
        prediction = predict_gru(trained, test)
        self.assertEqual(prediction.shape, (2, 3))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
