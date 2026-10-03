"""Read-only Genesis GridSat sequences and a small CNN--GRU classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .gridsat_gru import IMAGE_SIZE, load_crop


HISTORY_LENGTH = 9
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_MANIFEST = REPOSITORY_ROOT / "results/genesis_candidate_manifest.csv"
DEFAULT_STATE_DIR = REPOSITORY_ROOT.parent / "genesis_gridsat_state"
DEFAULT_CROP_ROOT = REPOSITORY_ROOT.parent / "genesis_gridsat_crops"


@dataclass(frozen=True)
class GenesisGridSatSamples:
    """Frozen Genesis candidates aligned with their nine historical crop paths."""

    candidates: pd.DataFrame
    image_paths: tuple[tuple[Path, ...], ...]


def _times(value: str) -> tuple[pd.Timestamp, ...]:
    times = tuple(pd.Timestamp(stamp, tz="UTC") for stamp in str(value).split("|"))
    if len(times) != HISTORY_LENGTH:
        raise ValueError(f"Genesis candidate must have {HISTORY_LENGTH} input times")
    return times


def load_genesis_samples(
    candidate_manifest: Path = DEFAULT_CANDIDATE_MANIFEST,
    state_dir: Path = DEFAULT_STATE_DIR,
    crop_root: Path = DEFAULT_CROP_ROOT,
) -> GenesisGridSatSamples:
    """Load only labeled candidates with nine latest-status saved crops, without writes."""
    candidates = pd.read_csv(candidate_manifest, dtype={"tcc_track_id": str})
    required = {"tcc_track_id", "input_times_utc", "label_24h", "split"}
    if missing := required - set(candidates):
        raise ValueError(f"Genesis candidate manifest lacks {sorted(missing)}")
    candidates["tcc_track_id"] = candidates.tcc_track_id.astype(str)
    candidates["_times"] = candidates.input_times_utc.map(_times)
    candidates["label_24h"] = pd.to_numeric(candidates.label_24h, errors="coerce")
    candidates = candidates.loc[candidates.label_24h.isin((0, 1))].copy()

    attempts_path = state_dir / "genesis_gridsat_acquisition_manifest.csv"
    attempts = pd.read_csv(attempts_path, dtype={"tcc_track_id": str})
    required_attempts = {"tcc_track_id", "time_utc", "status", "crop_path"}
    if missing := required_attempts - set(attempts):
        raise ValueError(f"Genesis acquisition manifest lacks {sorted(missing)}")
    attempts["tcc_track_id"] = attempts.tcc_track_id.astype(str)
    attempts["time_utc"] = pd.to_datetime(attempts.time_utc, utc=True, errors="raise")
    # The acquisition manifest is append-only; its last row is the latest status.
    attempts = attempts.drop_duplicates(["tcc_track_id", "time_utc"], keep="last")
    saved = attempts.loc[attempts.status.eq("saved") & attempts.crop_path.notna(), ["tcc_track_id", "time_utc", "crop_path"]]
    paths = {
        (row.tcc_track_id, row.time_utc): crop_root / Path(row.crop_path)
        for row in saved.itertuples(index=False)
    }

    kept_rows: list[int] = []
    histories: list[tuple[Path, ...]] = []
    for index, row in candidates.iterrows():
        history = tuple(paths.get((row.tcc_track_id, stamp)) for stamp in row["_times"])
        if all(path is not None and path.exists() for path in history):
            # Crop names are generated as split/...; retain the frozen split boundary.
            if all(path.relative_to(crop_root).parts[0] == row.split for path in history):
                kept_rows.append(index)
                histories.append(history)  # type: ignore[arg-type]
    kept = candidates.loc[kept_rows].drop(columns="_times").reset_index(drop=True)
    return GenesisGridSatSamples(kept, tuple(histories))


def load_genesis_images(samples: GenesisGridSatSamples, limit: int | None = None) -> torch.Tensor:
    """Return N x 9 x 1 x 201 x 201 IR tensors in oldest-to-newest order."""
    histories = samples.image_paths[:limit]
    if not histories:
        raise ValueError("no complete Genesis candidates available")
    images = np.stack([[np.nan_to_num(load_crop(path), nan=0.0) for path in history] for history in histories])
    return torch.from_numpy(images[:, :, None].astype(np.float32))


class GenesisCNNGRU(nn.Module):
    """Encode each IR image, summarize its nine-frame history, then classify Genesis."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.gru = nn.GRU(16, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or tuple(images.shape[1:]) != (HISTORY_LENGTH, 1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected N x {HISTORY_LENGTH} x 1 x {IMAGE_SIZE} x {IMAGE_SIZE} images")
        features = self.cnn(images.flatten(0, 1)).unflatten(0, (images.shape[0], HISTORY_LENGTH))
        _, hidden = self.gru(features)
        return self.classifier(hidden[-1]).squeeze(1)
