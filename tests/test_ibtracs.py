"""Phase 1 rules for the IBTrACS loader and sample builder."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest

import pandas as pd

from src.ibtracs import (
    CORE_COLUMNS,
    FEATURE_NAMES,
    build_forecast_samples,
    filter_north_atlantic,
    load_ibtracs,
    storm_split_map,
)


def row(sid: str, time: str, *, season: int = 2000, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "SID": sid, "SEASON": season, "NAME": "TEST", "BASIN": "NA", "ISO_TIME": pd.Timestamp(time),
        "TRACK_TYPE": "main", "USA_LAT": 10.0, "USA_LON": -50.0, "USA_WIND": 40.0,
        "USA_PRES": 1000.0, "USA_AGENCY": "hurdat_atl", "USA_STATUS": "TS", "IFLAG": "O",
    }
    value.update(overrides)
    return value


def storm(sid: str, start: str, periods: int, *, season: int = 2000) -> list[dict[str, object]]:
    times = pd.date_range(start, periods=periods, freq="6h", tz="UTC")
    return [row(sid, str(time), season=season, USA_LAT=10 + index, USA_LON=-50 + index, USA_WIND=40 + index)
            for index, time in enumerate(times)]


class IbtracsPhaseOneTests(unittest.TestCase):
    def test_loader_skips_units_and_preserves_na_basin(self) -> None:
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=CORE_COLUMNS)
            writer.writeheader()
            writer.writerow(dict.fromkeys(CORE_COLUMNS, "units"))
            source = row("A", "2000-08-01T00:00:00Z")
            source["ISO_TIME"] = "2000-08-01 00:00:00"
            writer.writerow(source)
            path = handle.name
        try:
            loaded = load_ibtracs(path)
        finally:
            os.unlink(path)

        self.assertEqual(loaded.BASIN.iloc[0], "NA")
        self.assertEqual(loaded.ISO_TIME.iloc[0], pd.Timestamp("2000-08-01T00:00:00Z"))

    def test_filter_enforces_source_time_status_core_and_duplicate_rules(self) -> None:
        records = storm("A", "2000-08-01T00:00:00Z", 2)
        records += [
            row("B", "2000-08-01T03:00:00Z"),
            row("C", "2000-08-01T00:00:00Z", USA_AGENCY="other"),
            row("D", "2000-08-01T00:00:00Z", USA_STATUS="EX"),
            row("E", "2000-08-01T00:00:00Z", USA_WIND=None),
            row("F", "2000-08-01T00:00:00Z"),
            row("F", "2000-08-01T00:00:00Z", USA_WIND=45.0),
        ]
        filtered = filter_north_atlantic(pd.DataFrame(records, columns=CORE_COLUMNS))
        self.assertEqual(filtered.SID.tolist(), ["A", "A"])

    def test_windows_need_contiguous_history_and_a_future_target(self) -> None:
        records = storm("A", "2000-08-01T00:00:00Z", 10)
        records.append(row("B", "2000-08-01T00:00:00Z"))
        records += storm("B", "2000-08-01T12:00:00Z", 6)
        samples = build_forecast_samples(filter_north_atlantic(pd.DataFrame(records)), 6)

        # A has five valid origins; B becomes eligible only after its early gap.
        self.assertEqual(len(samples.metadata), 6)
        self.assertEqual(samples.metadata.groupby("SID").size().to_dict(), {"A": 5, "B": 1})
        self.assertEqual(samples.features.shape, (6, 5, len(FEATURE_NAMES)))
        self.assertEqual(samples.targets.shape, (6, 3))
        self.assertTrue((samples.history_times[:, -1] == samples.metadata.issue_time.to_numpy()).all())
        self.assertTrue((samples.history_times <= samples.metadata.issue_time.to_numpy()[:, None]).all())

    def test_future_values_are_targets_not_history_features(self) -> None:
        data = filter_north_atlantic(pd.DataFrame(storm("A", "2000-08-01T00:00:00Z", 6)))
        samples = build_forecast_samples(data, 6)

        self.assertEqual(len(samples.metadata), 1)
        self.assertEqual(samples.features[0, -1, 2], 44.0)
        self.assertEqual(samples.targets[0, 2], 1.0)

    def test_twelve_hour_horizon_uses_the_same_history_and_direct_future_target(self) -> None:
        data = filter_north_atlantic(pd.DataFrame(storm("A", "2000-08-01T00:00:00Z", 7)))
        samples = build_forecast_samples(data, 12)

        self.assertEqual(len(samples.metadata), 1)
        self.assertEqual(samples.metadata.target_time.iloc[0] - samples.metadata.issue_time.iloc[0], pd.Timedelta(hours=12))
        self.assertTrue((samples.history_times <= samples.metadata.issue_time.to_numpy()[:, None]).all())

    def test_storm_split_is_atomic_and_chronological(self) -> None:
        data = pd.DataFrame([
            row("OLD", "1999-08-01T00:00:00Z", season=1999),
            row("CROSS", "2000-08-01T00:00:00Z", season=2000),
            row("CROSS", "2001-08-01T00:00:00Z", season=2001),
            row("NEW", "2003-08-01T00:00:00Z", season=2003),
        ])
        splits = storm_split_map(data, train_end_year=2000, validation_end_year=2001)
        self.assertEqual(splits.to_dict(), {"OLD": "train", "CROSS": "train", "NEW": "test"})

    def test_history_length_changes_required_past_states_not_future_access(self) -> None:
        data = filter_north_atlantic(pd.DataFrame(storm("A", "2000-08-01T00:00:00Z", 10)))
        short = build_forecast_samples(data, 6, history_steps=2)
        long = build_forecast_samples(data, 6, history_steps=9)
        self.assertEqual(len(short.metadata), 8)
        self.assertEqual(len(long.metadata), 1)
        self.assertTrue((long.history_times <= long.metadata.issue_time.to_numpy()[:, None]).all())


if __name__ == "__main__":
    unittest.main()
