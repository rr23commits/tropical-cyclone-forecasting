"""Genesis GridSat adapter and CNN--GRU shape tests with local synthetic crops."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.genesis_gridsat_gru import GenesisCNNGRU, load_genesis_images, load_genesis_samples


class GenesisGridSatGRUTests(unittest.TestCase):
    def test_adapter_uses_latest_saved_status_for_all_nine_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range("2000-01-01", periods=9, freq="3h", tz="UTC")
            names = []
            for number, stamp in enumerate(times):
                name = f"train/track-1-{number}.npz"
                path = root / "crops" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, image=np.full((201, 201), number, dtype=np.float32))
                names.append(name)
            pd.DataFrame([{"tcc_track_id": "track-1", "input_times_utc": "|".join(stamp.isoformat() for stamp in times), "label_24h": 1, "split": "train"},
                          {"tcc_track_id": "track-2", "input_times_utc": "|".join(stamp.isoformat() for stamp in times), "label_24h": 0, "split": "train"}]).to_csv(root / "candidates.csv", index=False)
            rows = [{"tcc_track_id": "track-1", "time_utc": stamp.isoformat(), "status": "failed", "crop_path": name} for stamp, name in zip(times, names)]
            rows += [{"tcc_track_id": "track-1", "time_utc": stamp.isoformat(), "status": "saved", "crop_path": name} for stamp, name in zip(times, names)]
            pd.DataFrame(rows).to_csv(root / "state.csv", index=False)
            state = root / "state"
            state.mkdir()
            (root / "state.csv").replace(state / "genesis_gridsat_acquisition_manifest.csv")
            samples = load_genesis_samples(root / "candidates.csv", state, root / "crops")
            self.assertEqual(samples.candidates.tcc_track_id.tolist(), ["track-1"])
            self.assertEqual(tuple(load_genesis_images(samples).shape), (1, 9, 1, 201, 201))

    def test_model_returns_one_logit_per_candidate_and_backpropagates(self) -> None:
        model = GenesisCNNGRU(hidden_size=4)
        logits = model(torch.zeros((2, 9, 1, 201, 201)))
        self.assertEqual(tuple(logits.shape), (2,))
        F.binary_cross_entropy_with_logits(logits, torch.tensor([0.0, 1.0])).backward()
        self.assertIsNotNone(model.classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()
