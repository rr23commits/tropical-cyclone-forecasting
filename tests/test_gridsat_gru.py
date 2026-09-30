"""Focused GridSat CNN+GRU tests using only synthetic local crops."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.gridsat_gru import (
    GridSatGRUForecast, ImageScaler, join_gridsat_samples, predict_gridsat_gru,
    split_gridsat_samples, train_select_gridsat_gru,
)
from src.gru import GRUSettings
from src.ibtracs import ForecastSamples


def samples(count: int = 4) -> ForecastSamples:
    issue_times = pd.date_range("2010-08-01", periods=count, freq="6h", tz="UTC")
    features = np.arange(count * 5 * 9, dtype=float).reshape(count, 5, 9) / 10.0
    targets = np.column_stack((features[:, -1, 0], features[:, -1, 1], features[:, -1, 2]))
    metadata = pd.DataFrame({
        "SID": [f"S{index}" for index in range(count)], "issue_time": issue_times,
        "target_time": issue_times + pd.Timedelta(hours=6), "issue_lat": np.zeros(count),
        "issue_lon": np.zeros(count), "issue_wind": np.zeros(count), "target_lat": np.zeros(count),
        "target_lon": np.zeros(count), "target_wind": np.zeros(count),
    })
    return ForecastSamples(features, targets, metadata, np.empty((count, 5), dtype=object), 6)


def write_crop(root: Path, name: str, value: float) -> str:
    relative = Path("crops") / f"{name}.npz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((201, 201), value, dtype=np.float16)
    image[0, 0] = np.nan
    np.savez_compressed(path, image=image)
    return str(relative)


class GridSatGRUTests(unittest.TestCase):
    def test_join_uses_utc_keys_and_requires_saved_existing_crops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = samples(3)
            crop = write_crop(root, "first", 200.0)
            pd.DataFrame([
                {"SID": "S0", "issue_time": "2010-08-01 00:00:00+00:00", "status": "saved", "crop_path": crop},
                {"SID": "S1", "issue_time": "2010-08-01 11:30:00+05:30", "status": "excluded", "crop_path": crop},
                {"SID": "S2", "issue_time": "2010-08-01 12:00:00Z", "status": "saved", "crop_path": "crops/missing.npz"},
            ]).to_csv(root / "manifest.csv", index=False)
            joined = join_gridsat_samples(data, root)
            self.assertEqual(joined.samples.metadata.SID.tolist(), ["S0"])
            self.assertEqual(joined.crop_paths, (root / crop,))

    def test_image_scaler_uses_training_crops_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / write_crop(root, "train", 200.0)
            validation = root / write_crop(root, "validation", 300.0)
            scaler = ImageScaler.fit((train,))
            self.assertAlmostEqual(scaler.mean, 200.0)
            self.assertGreater(scaler.transform(np.full((201, 201), 300.0))[0, 0], 90.0)
            self.assertEqual(scaler.transform(np.array([[np.nan]], dtype=np.float32))[0, 0], 0.0)

    def test_model_and_training_return_three_direct_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = samples()
            rows = []
            for index, row in data.metadata.iterrows():
                rows.append({"SID": row.SID, "issue_time": row.issue_time.isoformat(), "status": "saved", "crop_path": write_crop(root, str(index), 200.0 + index)})
            pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
            joined = join_gridsat_samples(data, root)
            groups = split_gridsat_samples(joined, pd.Series({"S0": "train", "S1": "train", "S2": "validation", "S3": "test"}))
            model = GridSatGRUForecast(9, 4)
            self.assertEqual(tuple(model(torch.zeros((2, 5, 9)), torch.zeros((2, 1, 201, 201))).shape), (2, 3))
            trained = train_select_gridsat_gru(groups["train"], groups["validation"], GRUSettings(hidden_sizes=(4,), max_epochs=2, patience=1, batch_size=2))
            prediction = predict_gridsat_gru(trained, groups["test"])
            self.assertEqual(prediction.shape, (1, 3))
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
