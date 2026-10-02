"""Focused resumability tests for Genesis/TCC GridSat crop acquisition."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src import acquire_genesis_gridsat as acquisition


class GenesisGridSatAcquireTests(unittest.TestCase):
    def test_requests_and_one_source_download_cover_multiple_tcc_positions(self) -> None:
        candidates = pd.DataFrame({"tcc_track_id": ["track-1", "track-2"], "split": ["train", "train"]})
        points = pd.DataFrame({
            "tcc_track_id": ["track-1", "track-2"],
            "time_utc": pd.to_datetime(["2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z"]),
            "tcc_lat": [10.0, 11.0], "tcc_lon": [20.0, 21.0],
        })
        with patch.object(acquisition, "cohort", return_value=candidates), \
             patch.object(acquisition, "_position_rows", return_value=points), \
             patch.object(acquisition, "source_name", return_value="source.nc"), \
             patch.object(acquisition, "source_url", return_value="https://example.test/source.nc"), \
             patch.object(acquisition, "_crop_name", side_effect=["train/a.npz", "train/b.npz"]):
            records = acquisition.request_records(pd.DataFrame(), pd.DataFrame())

        self.assertEqual(len(records), 2)
        self.assertEqual(records.source_timestamp_utc.nunique(), 1)

        def save_crop(field, lat, lon, row, path):
            if row.issue_lat == 11.0:
                return {"status": "excluded", "exclusion_reason": "invalid_crop_geometry", "height": 0,
                        "width": 0, "valid_fraction": 0.0, "saved_crop_bytes": 0}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"crop")
            return {"status": "saved", "exclusion_reason": "", "height": 201, "width": 201,
                    "valid_fraction": 1.0, "saved_crop_bytes": 4}

        response = MagicMock()
        response.read.side_effect = [b"source", b""]
        opener = MagicMock()
        opener.return_value.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(acquisition, "urlopen", opener), \
             patch.object(acquisition, "source_url", return_value="https://example.test/source.nc"), \
             patch.object(acquisition, "load_field", return_value=(None, None, None)) as load, \
             patch.object(acquisition, "write_crop", side_effect=save_crop) as crop:
            root = Path(directory) / "crops"
            attempts = Path(directory) / "state" / "acquisition.csv"
            summary = acquisition.acquire(records, root, attempts, max_source_files=0)
            self.assertEqual(summary, {"unique_source_timestamps": 1, "saved": 1, "excluded": 1, "failed": 0})
            self.assertEqual(opener.call_count, 1)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(crop.call_count, 2)
            saved = pd.read_csv(attempts)
            self.assertEqual(saved.columns.tolist(), list(acquisition.ATTEMPT_FIELDS))
            self.assertEqual(saved.loc[saved.status.eq("excluded"), "error"].item(), "invalid_crop_geometry")

            again = acquisition.acquire(records, root, attempts, max_source_files=0)
            self.assertEqual(again["unique_source_timestamps"], 0)
            self.assertEqual(opener.call_count, 1)


if __name__ == "__main__":
    unittest.main()
