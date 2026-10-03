#!/usr/bin/env python3
"""Write a read-only majority-class comparison for finalized Genesis results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def majority_metrics(positive: int, negative: int) -> dict[str, float | None]:
    """Metrics for the requested test-majority constant classifier."""
    total = positive + negative
    if total == 0:
        raise ValueError("test split has no candidates")
    majority = max(positive, negative)
    if positive >= negative:
        precision, recall = positive / total, 1.0
    else:
        precision, recall = 0.0, 0.0
    return {
        "accuracy": majority / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": None,
    }


def comparison(metrics: dict[str, object]) -> dict[str, object]:
    """Combine saved finalized CNN--GRU test metrics with the constant baseline."""
    counts = metrics.get("class_counts", {}).get("test", {})  # type: ignore[union-attr]
    cnn = metrics.get("test", {})
    if not all(name in counts for name in ("positive", "negative")) or not all(name in cnn for name in METRICS):
        raise ValueError("metrics JSON lacks finalized test counts or CNN-GRU metrics")
    baseline = majority_metrics(int(counts["positive"]), int(counts["negative"]))
    differences = {
        name: None if baseline[name] is None or cnn[name] is None else abs(float(cnn[name]) - float(baseline[name]))
        for name in METRICS
    }
    return {
        "test_class_counts": {"positive": int(counts["positive"]), "negative": int(counts["negative"])},
        "majority_class_baseline": baseline,
        "cnn_gru": {name: cnn[name] for name in METRICS},
        "absolute_metric_differences": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/genesis_cnn_gru/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("results/genesis_cnn_gru/baselines.json"))
    args = parser.parse_args()
    report = comparison(json.loads(args.metrics.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
