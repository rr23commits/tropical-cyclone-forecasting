"""Genesis GridSat adapter and CNN--GRU shape tests with local synthetic crops."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.genesis_gridsat_gru import (
    DEFAULT_CANDIDATE_MANIFEST, DEFAULT_CROP_ROOT, DEFAULT_STATE_DIR,
    GenesisCNNGRU, GenesisGridSatSamples, GenesisImageDataset, REPOSITORY_ROOT,
    _times, load_genesis_images, load_genesis_samples, split_genesis_samples,
)
from src.gridsat_gru import ImageScaler
from src.run_genesis_gridsat_gru import fit_image_scaler, materialize_crops


class GenesisGridSatGRUTests(unittest.TestCase):
    def test_defaults_resolve_repo_cohort_and_sibling_external_data(self) -> None:
        self.assertEqual(DEFAULT_CANDIDATE_MANIFEST, REPOSITORY_ROOT / "results/genesis_candidate_manifest.csv")
        self.assertEqual(DEFAULT_STATE_DIR, REPOSITORY_ROOT.parent / "genesis_gridsat_state")
        self.assertEqual(DEFAULT_CROP_ROOT, REPOSITORY_ROOT.parent / "genesis_gridsat_crops")

    def test_split_isolation_and_dataset_construction(self) -> None:
        samples = GenesisGridSatSamples(
            pd.DataFrame({"split": ["train", "validation", "test"], "label_24h": [0, 1, 0]}),
            ((Path("train.npz"),), (Path("validation.npz"),), (Path("test.npz"),)),
        )
        groups = split_genesis_samples(samples)
        self.assertEqual({split: group.candidates.split.tolist() for split, group in groups.items()},
                         {"train": ["train"], "validation": ["validation"], "test": ["test"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.npz"
            np.savez(path, image=np.zeros((201, 201), dtype=np.float32))
            dataset = GenesisImageDataset(GenesisGridSatSamples(
                pd.DataFrame({"split": ["train"], "label_24h": [1]}), ((path,) * 9,)
            ))
            images, label = dataset[0]
            self.assertEqual(tuple(images.shape), (9, 1, 201, 201))
            self.assertEqual(label.item(), 1.0)

    def test_local_crop_cache_and_observable_scaler_match_existing_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "drive" / "train" / "crop.npz"
            source.parent.mkdir(parents=True)
            np.savez(source, image=np.full((201, 201), 200.0, dtype=np.float32))
            samples = GenesisGridSatSamples(
                pd.DataFrame({"split": ["train"], "label_24h": [1]}), ((source,) * 9,)
            )
            messages: list[str] = []
            cached = materialize_crops({"train": samples}, root / "drive", root / "local", 1, messages.append)["train"]
            self.assertTrue(all(path.exists() and root / "local" in path.parents for path in cached.image_paths[0]))
            self.assertIn("progress: 1/1 crops (1 copied, 0 cached)", messages)
            paths = tuple(path for history in cached.image_paths for path in history)
            scaler = fit_image_scaler(paths, 1, messages.append)
            expected = ImageScaler.fit(paths)
            self.assertEqual(scaler, expected)
            self.assertIn("progress: 9/9 training crops", messages)

    def test_parser_and_adapter_handle_real_format_full_cohort(self) -> None:
        value = "|".join(f"1982-01-{day:02d}T{hour:02d}:00:00Z" for day, hour in ((2, 18), (2, 21), (3, 0), (3, 3), (3, 6), (3, 9), (3, 12), (3, 15), (3, 18)))
        self.assertEqual(_times(value), tuple(pd.Timestamp(stamp, tz="UTC") for stamp in value.split("|")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"tcc_track_id": range(1276), "input_times_utc": [value] * 1276,
                          "label_24h": [0] * 1276, "split": ["train"] * 1276}).to_csv(root / "candidates.csv", index=False)
            state = root / "state"
            state.mkdir()
            pd.DataFrame(columns=["tcc_track_id", "time_utc", "status", "crop_path"]).to_csv(
                state / "genesis_gridsat_acquisition_manifest.csv", index=False
            )
            self.assertEqual(len(load_genesis_samples(root / "candidates.csv", state, root / "crops").candidates), 0)

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
