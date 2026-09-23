import unittest

import pandas as pd

from src.monthly_activity_ensemble import blend_forecast, select_blend_weight


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "month": pd.date_range("2016-01-01", periods=4, freq="MS"),
            "actual": [1.0, 2.0, 3.0, 4.0],
            "split": ["validation", "validation", "test", "test"],
            "xgboost": [1.0, 2.0, 20.0, 20.0],
            "prophet": [3.0, 4.0, 30.0, 30.0],
        })

    def test_weight_selection_uses_validation_only(self):
        selected = select_blend_weight(self.frame)
        changed = self.frame.copy()
        changed.loc[changed.split == "test", "xgboost"] = -999
        changed.loc[changed.split == "test", "prophet"] = 999
        self.assertEqual(selected, select_blend_weight(changed))
        self.assertEqual(selected, 1.0)

    def test_blend_uses_xgboost_weight_and_complement(self):
        result = blend_forecast(self.frame.iloc[:2], 0.25)
        self.assertEqual(result.iloc[0], 2.5)
        self.assertEqual(result.iloc[1], 3.5)

    def test_validation_only_selection_requires_validation_rows(self):
        with self.assertRaises(ValueError):
            select_blend_weight(self.frame[self.frame.split == "test"])


if __name__ == "__main__":
    unittest.main()
