import unittest

import pandas as pd

from src.manifest_genesis_era5 import build_manifest, validate_manifest


class GenesisEra5ManifestTests(unittest.TestCase):
    def setUp(self):
        start = pd.Timestamp("2020-01-01T00:00:00Z")
        position_rows = []
        candidate_rows = []
        for index in range(1284):
            stamp = start + pd.Timedelta(hours=3 * index)
            position_rows.append({
                "tcc_track_id": "track-1", "time_utc": stamp.isoformat(), "tcc_lat": 10.0,
                "tcc_lon": 179.8 if index % 2 == 0 else -179.8,
            })
            if index >= 8:
                inputs = [start + pd.Timedelta(hours=3 * (index - offset)) for offset in range(8, -1, -1)]
                candidate_rows.append({
                    "tcc_track_id": "track-1", "basin": "WP", "year": stamp.year,
                    "issue_time_utc": stamp.isoformat(), "current_lat": 10.0,
                    "current_lon": 179.8 if index % 2 == 0 else -179.8,
                    "label_24h": index % 2, "label_48h": (index + 1) % 2,
                    "input_times_utc": "|".join(value.isoformat() for value in inputs),
                    "split": "train",
                })
        self.manifest = pd.DataFrame(candidate_rows)
        self.positions = pd.DataFrame(position_rows)

    def test_overlapping_histories_deduplicate_and_dateline_areas_stay_regional(self):
        document = build_manifest(self.manifest, self.positions)
        summary = document["estimates"]["request_counts"]
        self.assertEqual(summary["pressure"], summary["sst"])
        self.assertEqual(document["cohort"]["input_references_before_deduplication"], 1276 * 9)
        self.assertEqual(document["cohort"]["deduplicated_track_time_positions"], 1284)
        self.assertEqual(document["cohort"]["overlapping_references_removed"], 1276 * 9 - 1284)
        pressure = next(row for row in document["requests"] if row["product"] == "pressure")
        north, west, south, east = pressure["payload"]["area"]
        self.assertGreater(west, east)
        self.assertLess(north - south, 10)
        self.assertEqual(pressure["payload"]["variable"], [
            "u_component_of_wind", "v_component_of_wind", "temperature",
            "specific_humidity", "vertical_velocity",
        ])
        self.assertEqual(pressure["payload"]["pressure_level"], ["200", "500", "700", "850"])
        crop = document["crop_centers"][0]
        self.assertEqual(crop["grid_center_longitude"], 179.75)
        self.assertGreater(crop["crop_area_north_west_south_east"][1], crop["crop_area_north_west_south_east"][3])
        validate_manifest(document, self.manifest, self.positions)

    def test_validation_rejects_changed_predictor_or_cohort(self):
        document = build_manifest(self.manifest, self.positions)
        document["requests"][0]["payload"]["variable"] = ["relative_humidity"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_manifest(document, self.manifest, self.positions)

        changed = self.manifest.copy()
        changed.loc[0, "label_24h"] = 1 - changed.loc[0, "label_24h"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_manifest(build_manifest(self.manifest, self.positions), changed, self.positions)


if __name__ == "__main__":
    unittest.main()
