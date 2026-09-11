"""Compact, direct GRU forecasting for the existing IBTrACS sample sequences."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .ibtracs import ForecastSamples


@dataclass(frozen=True)
class GRUSettings:
    """Small fixed training budget; only hidden width is selected on validation data."""

    hidden_sizes: tuple[int, ...] = (16, 32)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 15
    seed: int = 42


@dataclass(frozen=True)
class SequenceScaler:
    """Per-feature and per-target training-set normalization statistics."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    @classmethod
    def fit(cls, samples: ForecastSamples) -> "SequenceScaler":
        feature_mean = samples.features.mean(axis=(0, 1))
        feature_scale = samples.features.std(axis=(0, 1))
        feature_scale[feature_scale == 0] = 1.0
        target_mean = samples.targets.mean(axis=0)
        target_scale = samples.targets.std(axis=0)
        target_scale[target_scale == 0] = 1.0
        return cls(feature_mean, feature_scale, target_mean, target_scale)

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.feature_mean) / self.feature_scale).astype(np.float32)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return ((targets - self.target_mean) / self.target_scale).astype(np.float32)

    def inverse_targets(self, targets: np.ndarray) -> np.ndarray:
        return targets * self.target_scale + self.target_mean


class GRUForecast(nn.Module):
    """One-layer GRU followed by a three-value direct forecast head."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.output(hidden[-1])


@dataclass(frozen=True)
class TrainedGRU:
    """Selected model plus the training-only scaler needed for future inference."""

    model: GRUForecast
    scaler: SequenceScaler
    hidden_size: int
    epochs_trained: int
    validation_mse: float
    settings: GRUSettings


def set_seed(seed: int) -> None:
    """Set CPU training randomness once per candidate model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _validation_mse(model: GRUForecast, features: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(features) - targets) ** 2).item())


def _train_candidate(
    train: ForecastSamples, validation: ForecastSamples, scaler: SequenceScaler, hidden_size: int, settings: GRUSettings
) -> TrainedGRU:
    set_seed(settings.seed)
    model = GRUForecast(train.features.shape[-1], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    train_features = torch.from_numpy(scaler.transform_features(train.features))
    train_targets = torch.from_numpy(scaler.transform_targets(train.targets))
    validation_features = torch.from_numpy(scaler.transform_features(validation.features))
    validation_targets = torch.from_numpy(scaler.transform_targets(validation.targets))
    generator = torch.Generator().manual_seed(settings.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_features), generator=generator).split(settings.batch_size):
            optimizer.zero_grad()
            loss = torch.mean((model(train_features[indices]) - train_targets[indices]) ** 2)
            loss.backward()
            optimizer.step()
        validation_loss = _validation_mse(model, validation_features, validation_targets)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break

    model.load_state_dict(best_state)
    return TrainedGRU(model, scaler, hidden_size, best_epoch, best_loss, settings)


def train_select_gru(train: ForecastSamples, validation: ForecastSamples, settings: GRUSettings = GRUSettings()) -> TrainedGRU:
    """Fit compact candidates on training rows and select only by validation MSE."""
    if not len(train.metadata) or not len(validation.metadata):
        raise ValueError("training and validation samples must both be non-empty")
    scaler = SequenceScaler.fit(train)
    candidates = [
        _train_candidate(train, validation, scaler, hidden_size, settings)
        for hidden_size in settings.hidden_sizes
    ]
    return min(candidates, key=lambda candidate: candidate.validation_mse)


def predict_gru(model: TrainedGRU, samples: ForecastSamples) -> np.ndarray:
    """Return unscaled direct displacement and wind-change predictions."""
    model.model.eval()
    features = torch.from_numpy(model.scaler.transform_features(samples.features))
    with torch.no_grad():
        standardized = model.model(features).cpu().numpy()
    return model.scaler.inverse_targets(standardized)
