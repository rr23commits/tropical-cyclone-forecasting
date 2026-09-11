"""Simple direct baselines for tropical-cyclone displacement and wind change."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ibtracs import ForecastSamples


def persistence(samples: ForecastSamples) -> np.ndarray:
    """Predict zero displacement and zero wind change from the issue state."""
    return np.zeros((len(samples.metadata), 3), dtype=float)


def constant_motion(samples: ForecastSamples) -> np.ndarray:
    """Extrapolate the final observed six-hour motion and retain current wind."""
    steps = samples.horizon_hours / 6.0
    prediction = np.zeros((len(samples.metadata), 3), dtype=float)
    # Feature columns 3 and 4 are the final north/east six-hour displacement.
    prediction[:, :2] = samples.features[:, -1, 3:5] * steps
    return prediction


def flattened_features(samples: ForecastSamples) -> np.ndarray:
    """Flatten one historical sequence per forecast origin for the lagged baseline."""
    return samples.features.reshape(len(samples.features), -1)


@dataclass(frozen=True)
class RidgeModel:
    """Training-only standardization and multi-output ridge coefficients."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.target_mean


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> RidgeModel:
    """Fit multi-output ridge regression using only the provided training rows."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    standardized = (features - feature_mean) / feature_scale
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    penalty = alpha * np.eye(standardized.shape[1])
    coefficients = np.linalg.solve(standardized.T @ standardized + penalty, standardized.T @ centered_targets)
    return RidgeModel(feature_mean, feature_scale, target_mean, coefficients, alpha)


def select_ridge(
    train: ForecastSamples, validation: ForecastSamples, alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
) -> RidgeModel:
    """Choose alpha on validation target RMSE standardized by training-target variation."""
    train_features = flattened_features(train)
    validation_features = flattened_features(validation)
    target_scale = train.targets.std(axis=0)
    target_scale[target_scale == 0] = 1.0

    candidates = []
    for alpha in alphas:
        model = fit_ridge(train_features, train.targets, alpha)
        residual = (model.predict(validation_features) - validation.targets) / target_scale
        candidates.append((float(np.sqrt(np.mean(residual ** 2))), model))
    return min(candidates, key=lambda candidate: candidate[0])[1]
