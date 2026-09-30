"""Direct GridSat IR plus track-history GRU forecasting on matched origins."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluation import subset_samples
from .gru import GRUSettings, SequenceScaler, set_seed
from .ibtracs import ForecastSamples


IMAGE_SIZE = 201


@dataclass(frozen=True)
class GridSatSamples:
    """Forecast rows and their one-to-one, issue-time GridSat crop paths."""

    samples: ForecastSamples
    crop_paths: tuple[Path, ...]


def _crop_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("manifest crop_path must be relative to the dataset root")
    return dataset_root / path


def join_gridsat_samples(samples: ForecastSamples, dataset_root: Path) -> GridSatSamples:
    """Keep only saved, existing crops matched exactly by SID and UTC issue time."""
    manifest_path = dataset_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, usecols=["SID", "issue_time", "status", "crop_path"])
    manifest["issue_time"] = pd.to_datetime(manifest["issue_time"], utc=True, errors="raise")
    manifest = manifest.loc[manifest.status.eq("saved") & manifest.crop_path.notna()].copy()
    if manifest.duplicated(["SID", "issue_time"]).any():
        raise ValueError("saved GridSat manifest rows must be unique by SID and issue_time")
    manifest["crop_path"] = manifest.crop_path.map(lambda value: _crop_path(dataset_root, value))
    manifest = manifest.loc[manifest.crop_path.map(Path.exists)].copy()

    keys = samples.metadata.loc[:, ["SID", "issue_time"]].copy()
    keys["issue_time"] = pd.to_datetime(keys["issue_time"], utc=True, errors="raise")
    if keys.duplicated(["SID", "issue_time"]).any():
        raise ValueError("forecast samples must be unique by SID and issue_time")
    keys["_sample_index"] = np.arange(len(keys))
    matched = keys.merge(manifest.loc[:, ["SID", "issue_time", "crop_path"]], on=["SID", "issue_time"], how="inner", validate="one_to_one")
    matched = matched.sort_values("_sample_index")
    indices = matched._sample_index.to_numpy(dtype=int)
    return GridSatSamples(subset_samples(samples, indices), tuple(matched.crop_path.tolist()))


def subset_gridsat_samples(samples: GridSatSamples, indices: np.ndarray) -> GridSatSamples:
    """Subset samples and crop paths together so their row alignment cannot drift."""
    return GridSatSamples(
        subset_samples(samples.samples, indices),
        tuple(samples.crop_paths[index] for index in indices),
    )


def split_gridsat_samples(samples: GridSatSamples, split_map: pd.Series) -> dict[str, GridSatSamples]:
    """Apply the existing atomic SID split without changing crop alignment."""
    labels = samples.samples.metadata["SID"].map(split_map)
    if labels.isna().any():
        raise ValueError("every sample storm must have a split label")
    return {
        split: subset_gridsat_samples(samples, np.flatnonzero(labels.eq(split).to_numpy()))
        for split in ("train", "validation", "test")
    }


def load_crop(path: Path) -> np.ndarray:
    """Load one extractor-produced IR crop, retaining missing pixels for normalization."""
    with np.load(path) as payload:
        if "image" not in payload:
            raise ValueError(f"GridSat crop has no image array: {path}")
        image = np.asarray(payload["image"], dtype=np.float32)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"GridSat crop must have shape {(IMAGE_SIZE, IMAGE_SIZE)}: {path}")
    return image


@dataclass(frozen=True)
class ImageScaler:
    """One training-only IR mean/scale; missing pixels become the normalized mean."""

    mean: float
    scale: float

    @classmethod
    def fit(cls, crop_paths: tuple[Path, ...]) -> "ImageScaler":
        if not crop_paths:
            raise ValueError("cannot fit an image scaler without training crops")
        total = squared_total = 0.0
        count = 0
        for path in crop_paths:
            values = load_crop(path)
            valid = values[np.isfinite(values)]
            total += float(valid.sum(dtype=np.float64))
            squared_total += float(np.square(valid, dtype=np.float64).sum())
            count += len(valid)
        if not count:
            raise ValueError("training GridSat crops contain no finite IR pixels")
        mean = total / count
        variance = max(squared_total / count - mean ** 2, 0.0)
        return cls(mean, max(float(np.sqrt(variance)), 1.0))

    def transform(self, image: np.ndarray) -> np.ndarray:
        standardized = (image - self.mean) / self.scale
        # Saved crops may retain up to 10% missing pixels. Mean imputation maps
        # them to zero without deriving a value from validation or test imagery.
        return np.nan_to_num(standardized, nan=0.0).astype(np.float32)


def _images(paths: tuple[Path, ...], scaler: ImageScaler) -> torch.Tensor:
    array = np.stack([scaler.transform(load_crop(path)) for path in paths])[:, None, :, :]
    return torch.from_numpy(array)


class GridSatGRUForecast(nn.Module):
    """Small current-image CNN fused with the existing-style recurrent embedding."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fusion = nn.Sequential(nn.Linear(hidden_size + 16, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, sequence: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.fusion(torch.cat((hidden[-1], self.cnn(image)), dim=1))


@dataclass(frozen=True)
class TrainedGridSatGRU:
    model: GridSatGRUForecast
    sequence_scaler: SequenceScaler
    image_scaler: ImageScaler
    hidden_size: int
    epochs_trained: int
    validation_mse: float
    settings: GRUSettings


def _validation_mse(model: GridSatGRUForecast, features: torch.Tensor, targets: torch.Tensor, paths: tuple[Path, ...], scaler: ImageScaler, batch_size: int) -> float:
    model.eval()
    squared_error = 0.0
    with torch.no_grad():
        for indices in torch.arange(len(paths)).split(batch_size):
            image = _images(tuple(paths[index] for index in indices.tolist()), scaler)
            residual = model(features[indices], image) - targets[indices]
            squared_error += float(torch.sum(residual ** 2).item())
    return squared_error / (len(paths) * targets.shape[1])


def _train_candidate(train: GridSatSamples, validation: GridSatSamples, sequence_scaler: SequenceScaler, image_scaler: ImageScaler, hidden_size: int, settings: GRUSettings) -> TrainedGridSatGRU:
    set_seed(settings.seed)
    model = GridSatGRUForecast(train.samples.features.shape[-1], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    train_features = torch.from_numpy(sequence_scaler.transform_features(train.samples.features))
    train_targets = torch.from_numpy(sequence_scaler.transform_targets(train.samples.targets))
    validation_features = torch.from_numpy(sequence_scaler.transform_features(validation.samples.features))
    validation_targets = torch.from_numpy(sequence_scaler.transform_targets(validation.samples.targets))
    generator = torch.Generator().manual_seed(settings.seed)
    best_state, best_loss, best_epoch, stale_epochs = copy.deepcopy(model.state_dict()), float("inf"), 0, 0

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_features), generator=generator).split(settings.batch_size):
            batch_paths = tuple(train.crop_paths[index] for index in indices.tolist())
            optimizer.zero_grad()
            loss = torch.mean((model(train_features[indices], _images(batch_paths, image_scaler)) - train_targets[indices]) ** 2)
            loss.backward()
            optimizer.step()
        validation_loss = _validation_mse(model, validation_features, validation_targets, validation.crop_paths, image_scaler, settings.batch_size)
        if validation_loss < best_loss:
            best_state, best_loss, best_epoch, stale_epochs = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break
    model.load_state_dict(best_state)
    return TrainedGridSatGRU(model, sequence_scaler, image_scaler, hidden_size, best_epoch, best_loss, settings)


def train_select_gridsat_gru(train: GridSatSamples, validation: GridSatSamples, settings: GRUSettings = GRUSettings()) -> TrainedGridSatGRU:
    """Select GRU width and checkpoint with training/validation rows only."""
    if not len(train.samples.metadata) or not len(validation.samples.metadata):
        raise ValueError("training and validation samples must both be non-empty")
    sequence_scaler = SequenceScaler.fit(train.samples)
    image_scaler = ImageScaler.fit(train.crop_paths)
    candidates = [_train_candidate(train, validation, sequence_scaler, image_scaler, hidden_size, settings) for hidden_size in settings.hidden_sizes]
    return min(candidates, key=lambda candidate: candidate.validation_mse)


def predict_gridsat_gru(model: TrainedGridSatGRU, samples: GridSatSamples) -> np.ndarray:
    """Predict unscaled direct targets without reading target values."""
    model.model.eval()
    features = torch.from_numpy(model.sequence_scaler.transform_features(samples.samples.features))
    predictions = []
    with torch.no_grad():
        for indices in torch.arange(len(samples.crop_paths)).split(model.settings.batch_size):
            paths = tuple(samples.crop_paths[index] for index in indices.tolist())
            predictions.append(model.model(features[indices], _images(paths, model.image_scaler)).cpu().numpy())
    return model.sequence_scaler.inverse_targets(np.concatenate(predictions))
