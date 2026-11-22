import unittest

import numpy as np
import pandas as pd
import torch

from src.monthly_activity_common import diebold_mariano, extended_metrics, mape_nonzero, season_of, smape
from src.monthly_activity_garch import candidate_grid, harmonics, interval_scores
from src.monthly_activity_rnn import baseline_grid, build_samples, tuned_candidates
from src.monthly_activity_transformer import PatchTST, candidates as patch_candidates, patchify
from src.monthly_activity_var import half_month_system
from src.monthly_activity_xgboost_tuned import candidates as xgb_candidates, feature_matrix


def make_series() -> pd.Series:
    index = pd.date_range("1980-01-01", "2025-12-01", freq="MS", tz="UTC")
    return pd.Series(np.arange(len(index), dtype=float), index=index)


class MetricTests(unittest.TestCase):
    def test_mape_ignores_zero_actual_months_and_reports_count(self):
        value, n = mape_nonzero(np.array([0.0, 2.0, 4.0]), np.array([1.0, 1.0, 5.0]))
        self.assertEqual(n, 2)
        self.assertAlmostEqual(value, (0.5 + 0.25) / 2 * 100)

    def test_smape_zero_zero_is_zero_error(self):
        self.assertEqual(smape(np.array([0.0, 0.0]), np.array([0.0, 0.0])), 0.0)

    def test_extended_metrics_rmse_mae(self):
        train = pd.Series([0.0, 1.0, 2.0, 3.0])
        m = extended_metrics(np.array([1.0, 3.0]), np.array([2.0, 1.0]), train)
        self.assertAlmostEqual(m["mae"], 1.5)
        self.assertAlmostEqual(m["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(m["bias"], -0.5)

    def test_diebold_mariano_prefers_better_forecast(self):
        rng = np.random.default_rng(0)
        y = rng.poisson(3, 200).astype(float)
        stat, p = diebold_mariano(y, y + rng.normal(0, 0.2, 200), y + rng.normal(0, 2.0, 200))
        self.assertLess(stat, 0)
        self.assertLess(p, 0.01)

    def test_season_phases_cover_all_months(self):
        self.assertEqual({season_of(m) for m in range(1, 13)}, {"off_season (Dec-May)", "early_season (Jun-Jul)", "peak_season (Aug-Oct)", "late_season (Nov)"})


class RecurrentWindowTests(unittest.TestCase):
    def test_windows_never_use_target_or_future(self):
        series = make_series()
        target = pd.Timestamp("2020-01-01", tz="UTC")
        pos = series.index.get_loc(target)
        changed = series.copy()
        changed.iloc[pos:] = 9999
        a = build_samples(series, pos, 12, True)
        b = build_samples(changed, pos, 12, True)
        for left, right in zip(a, b):
            self.assertTrue(np.array_equal(left, right))
        self.assertEqual(a[3][-1], series.iloc[pos - 1])          # last training target is the month before the forecast
        self.assertTrue(np.array_equal(a[4][0], series.iloc[pos - 12:pos].to_numpy(dtype=np.float32)))

    def test_candidates_reproducible_and_include_baseline(self):
        self.assertEqual(tuned_candidates("gru", 6), tuned_candidates("gru", 6))
        self.assertEqual(tuned_candidates("gru", 6)[0], baseline_grid("gru")[0])
        self.assertEqual(len(baseline_grid("gru")), 4)


class TransformerTests(unittest.TestCase):
    def test_patchify_and_forward_shapes(self):
        x = torch.randn(5, 24)
        self.assertEqual(patchify(x, 6, 3).shape, (5, 7, 6))
        model = PatchTST(24, 6, 3, 16, 2, 1, 0.1, extra=2)
        self.assertEqual(model(x, torch.randn(5, 2)).shape, (5, 1))

    def test_candidates_are_valid_patch_configs(self):
        for c in patch_candidates(8):
            self.assertEqual(c["d_model"] % c["heads"], 0)
            self.assertGreaterEqual((c["lookback"] - c["patch_len"]) // c["stride"] + 1, 1)


class GarchVarTests(unittest.TestCase):
    def test_harmonics_shape_and_period(self):
        months = pd.date_range("2000-01-01", periods=24, freq="MS", tz="UTC")
        h = harmonics(months, 3)
        self.assertEqual(h.shape, (24, 6))
        self.assertTrue(np.allclose(h[:12], h[12:]))

    def test_grid_has_arch_and_garch(self):
        grid = candidate_grid()
        self.assertTrue(any(c["q"] == 0 for c in grid) and any(c["q"] == 1 for c in grid))

    def test_interval_scores_coverage(self):
        scores = interval_scores(np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        self.assertEqual(scores["coverage_95"], 50.0)

    def test_half_month_system_sums_to_count(self):
        index = pd.date_range("1980-01-01", "2025-12-01", freq="MS", tz="UTC")
        audit = pd.DataFrame({"ISO_TIME": ["1980-08-03 00:00:00+00:00", "1980-08-20 06:00:00+00:00", "1980-09-16 00:00:00+00:00"],
                              "month": ["1980-08-01 00:00:00+00:00", "1980-08-01 00:00:00+00:00", "1980-09-01 00:00:00+00:00"]})
        counts = audit.groupby("month").size()
        series = pd.Series(0.0, index=index)
        series.loc[pd.Timestamp("1980-08-01", tz="UTC")] = 2.0
        series.loc[pd.Timestamp("1980-09-01", tz="UTC")] = 1.0
        system = half_month_system(series, audit)
        self.assertEqual(system.loc[pd.Timestamp("1980-08-01", tz="UTC"), "first_half"], 1.0)
        self.assertEqual(system.loc[pd.Timestamp("1980-08-01", tz="UTC"), "second_half"], 1.0)
        self.assertEqual(system.loc[pd.Timestamp("1980-09-01", tz="UTC"), "second_half"], 1.0)


class TunedXGBoostTests(unittest.TestCase):
    def test_features_use_only_past_values(self):
        series = make_series()
        target = pd.Timestamp("2020-08-01", tz="UTC")
        changed = series.copy()
        changed.loc[target:] = 12345.0
        for feature_set in ("base", "extended"):
            a = feature_matrix(series, feature_set).loc[target]
            b = feature_matrix(changed, feature_set).loc[target]
            self.assertTrue(a.equals(b), feature_set)

    def test_search_includes_anchor_and_is_reproducible(self):
        self.assertEqual(xgb_candidates(10), xgb_candidates(10))
        self.assertEqual(xgb_candidates(10)[0]["max_depth"], 1)


if __name__ == "__main__":
    unittest.main()
