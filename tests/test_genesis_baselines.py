"""Focused metrics tests for the read-only Genesis majority baseline."""

from __future__ import annotations

import unittest

from src.run_genesis_baselines import comparison, majority_metrics


class GenesisBaselineTests(unittest.TestCase):
    def test_positive_majority_baseline_and_comparison(self) -> None:
        baseline = majority_metrics(210, 101)
        self.assertAlmostEqual(baseline["accuracy"], 210 / 311)
        self.assertAlmostEqual(baseline["precision"], 210 / 311)
        self.assertEqual(baseline["recall"], 1.0)
        self.assertIsNone(baseline["roc_auc"])
        report = comparison({"class_counts": {"test": {"positive": 210, "negative": 101}}, "test": {
            "accuracy": 0.610932, "precision": 0.683128, "recall": 0.790476,
            "f1": 0.732892, "roc_auc": 0.561716,
        }})
        self.assertEqual(report["test_class_counts"], {"positive": 210, "negative": 101})
        self.assertIsNone(report["absolute_metric_differences"]["roc_auc"])
        self.assertGreater(report["absolute_metric_differences"]["accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
