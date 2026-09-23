import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.monthly_activity import (
    END,
    START,
    TRAIN_END,
    VALIDATION_END,
    _forecasts,
    build_monthly_series,
    candidate_orders,
    seasonal_naive,
    split_label,
)


def rows(items):
    frame = pd.DataFrame(items, columns=["SID", "SEASON", "NAME", "BASIN", "ISO_TIME", "TRACK_TYPE", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES", "USA_AGENCY", "USA_STATUS", "IFLAG"])
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], utc=True)
    return frame


class MonthlyActivityTests(unittest.TestCase):
    def test_counts_sid_once_at_first_retained_observation_and_fills_zeros(self):
        data = rows([
            ["A", 1980, "A", "NA", "1980-01-15 00:00", "main", 1, 1, 40, 990, "hurdat_atl", "TS", ""],
            ["A", 1980, "A", "NA", "1980-01-15 06:00", "main", 1, 1, 40, 990, "hurdat_atl", "TS", ""],
            ["B", 1980, "B", "NA", "1980-03-15 00:00", "main", 1, 1, 40, 990, "hurdat_atl", "TS", ""],
            ["B", 1980, "B", "NA", "1980-03-15 06:00", "main", 1, 1, 40, 990, "hurdat_atl", "TS", ""],
        ])
        series, audit = build_monthly_series(data)
        self.assertEqual(len(series), 552)
        self.assertEqual(series.loc["1980-01-01"], 1)
        self.assertEqual(series.loc["1980-02-01"], 0)
        self.assertEqual(series.loc["1980-03-01"], 1)
        self.assertEqual(audit.SID.nunique(), 2)

    def test_split_boundaries(self):
        self.assertEqual(split_label(TRAIN_END), "train")
        self.assertEqual(split_label(TRAIN_END + pd.offsets.MonthBegin(1)), "validation")
        self.assertEqual(split_label(VALIDATION_END), "validation")
        self.assertEqual(split_label(VALIDATION_END + pd.offsets.MonthBegin(1)), "test")

    def test_zero_is_observation(self):
        data = rows([["A", 2020, "A", "NA", "2020-01-01 00:00", "main", 1, 1, 40, 990, "hurdat_atl", "TS", ""]])
        series, _ = build_monthly_series(data)
        self.assertEqual(series.loc["2020-02-01"], 0)
        self.assertFalse(pd.isna(series.loc["2020-02-01"]))

    def test_lag_12_baseline_and_shared_forecast_timestamps(self):
        index = pd.date_range(START, END, freq="MS", tz="UTC")
        values = np.arange(len(index), dtype=float)
        series = pd.Series(values, index=index)
        self.assertEqual(seasonal_naive(values), values[-12])
        forecasts = [_forecasts(series, "seasonal_naive", "t-12"), _forecasts(series, "arima", (0, 1, 0))]
        self.assertListEqual(list(forecasts[0].index), list(forecasts[1].index))

    def test_forecast_does_not_use_current_or_future_value(self):
        index = pd.date_range(START, END, freq="MS", tz="UTC")
        base = pd.Series(np.ones(len(index)), index=index)
        changed = base.copy(); changed.loc["2020-01-01":] = 999
        first_base = _forecasts(base, "arima", (1, 1, 0)).loc["2019-12-01"]
        first_changed = _forecasts(changed, "arima", (1, 1, 0)).loc["2019-12-01"]
        self.assertEqual(first_base, first_changed)

    def test_arima_and_sarima_grids_include_ma_terms(self):
        orders = candidate_orders()
        self.assertTrue(any(order[2] > 0 for order in orders["arima"]))
        self.assertTrue(any(order[0][2] > 0 and order[1][2] > 0 for order in orders["sarima"]))

    def test_order_selection_only_requests_validation_history(self):
        index = pd.date_range(START, END, freq="MS", tz="UTC")
        series = pd.Series(np.ones(len(index)), index=index)
        seen = []

        def fake_forecasts(history, model, setting):
            seen.append(history.index[-1])
            return pd.Series(np.zeros(len(history.index[history.index > TRAIN_END])), index=history.index[history.index > TRAIN_END])

        from src import monthly_activity
        with patch.object(monthly_activity, "_forecasts", side_effect=fake_forecasts):
            monthly_activity._select_settings(series)
        self.assertTrue(seen)
        self.assertTrue(all(timestamp <= VALIDATION_END for timestamp in seen))


if __name__ == "__main__":
    unittest.main()
