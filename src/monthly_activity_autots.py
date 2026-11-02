"""AutoTS monthly storm-activity experiment.

AutoTS searches a small template pool on the training period once.  Its
templates are then scored with the same expanding one-step validation window
used by the other monthly models.  The validation winner is refit on the
history available at each forecast origin, so neither the test period nor a
fresh genetic search is used during held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .monthly_activity import (
    TRAIN_END,
    VALIDATION_END,
    _metrics,
    _svg,
    build_monthly_series,
    load_ibtracs,
    seasonal_naive,
    split_label,
)


RANDOM_SEED = 42
TEMPLATE_CANDIDATES = 3
# A single, univariate monthly series does not justify AutoTS's default broad
# search. The exported templates still retain AutoTS's model/transform choices.
SEARCH_OPTIONS: dict[str, object] = {
    "forecast_length": 1,
    "frequency": "MS",
    "max_generations": 1,
    "num_validations": 1,
    "models_to_validate": 0.25,
    "model_list": "superfast",
    "transformer_list": "superfast",
    "transformer_max_depth": 2,
    "ensemble": None,
    "no_negatives": True,
    "random_seed": RANDOM_SEED,
    "n_jobs": 1,
    "verbose": 0,
}


def count_forecast(value: object, fallback: float) -> float:
    """Return a finite nonnegative count, rounded half-up to a whole storm."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = fallback
    if not np.isfinite(numeric):
        numeric = fallback
    # Counts are reported as integer point forecasts; half-up avoids depending
    # on NumPy's banker's-rounding convention at exact .5 boundaries.
    return float(max(0, np.floor(numeric + 0.5)))


def history_frame(series: pd.Series, target_month: pd.Timestamp) -> pd.DataFrame:
    """Return the one-column AutoTS history that ends strictly before target."""
    history = series.loc[:target_month - pd.offsets.MonthBegin(1)]
    return pd.DataFrame({"storm_count": history.to_numpy(dtype=float)}, index=history.index.tz_localize(None))


def _autots_class() -> Any:
    try:
        from autots import AutoTS
    except ImportError as exc:  # keeps unrelated monthly commands importable
        raise RuntimeError("AutoTS is required: install dependencies with `pip install -r requirements.txt`.") from exc
    return AutoTS


def search_templates(train: pd.Series, output: Path) -> pd.DataFrame:
    """Run one bounded search on training data and export reusable templates."""
    AutoTS = _autots_class()
    model = AutoTS(**SEARCH_OPTIONS)
    model.fit(pd.DataFrame({"storm_count": train.to_numpy(dtype=float)}, index=train.index.tz_localize(None)))
    path = output / "autots_candidate_templates.csv"
    model.export_template(str(path), models="best", n=TEMPLATE_CANDIDATES, include_results=True)
    templates = pd.read_csv(path)
    if templates.empty:
        raise RuntimeError("AutoTS search exported no reusable model templates")
    return templates.head(TEMPLATE_CANDIDATES).reset_index(drop=True)


def _fixed_template_predict(series: pd.Series, target_month: pd.Timestamp, template: pd.DataFrame) -> float:
    """Refit one selected template on available history and make one forecast."""
    AutoTS = _autots_class()
    # import_best_model + fit_data is AutoTS's production refit path. Unlike
    # fit after import_template, it does not revalidate or evolve candidates.
    model = AutoTS(**{**SEARCH_OPTIONS, "max_generations": 0, "num_validations": 0})
    model.import_best_model(template, enforce_model_list=False, include_ensemble=False)
    model.fit_data(history_frame(series, target_month))
    prediction = model.predict(forecast_length=1, fail_on_forecast_nan=True)
    return float(prediction.forecast.iloc[0, 0])


def expanding_forecast(series: pd.Series, target_months: pd.DatetimeIndex, template: pd.DataFrame, failures: list[dict[str, str]] | None = None) -> pd.Series:
    """One-step forecasts, using only observations strictly before each target."""
    predictions = []
    for month in target_months:
        fallback = seasonal_naive(series.loc[:month - pd.offsets.MonthBegin(1)].to_numpy())
        try:
            value = _fixed_template_predict(series, month, template)
        except Exception as exc:  # an unsupported imported candidate must not stop the experiment
            value = fallback
            if failures is not None:
                failures.append({"month": str(month), "error": f"{type(exc).__name__}: {exc}"})
        predictions.append(count_forecast(value, fallback))
    return pd.Series(predictions, index=target_months, name="forecast")


def select_template(series: pd.Series, templates: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, str]]]:
    """Select only from validation one-step MAE; test observations are untouched."""
    validation = series.index[(series.index > TRAIN_END) & (series.index <= VALIDATION_END)]
    scores: list[dict[str, object]] = []
    selection_failures: list[dict[str, str]] = []
    for candidate_id in range(len(templates)):
        failures: list[dict[str, str]] = []
        prediction = expanding_forecast(series, validation, templates.iloc[[candidate_id]], failures)
        scores.append({"candidate": candidate_id, "validation_mae": float(np.mean(np.abs(prediction - series.loc[validation]))), "fallback_months": len(failures)})
        selection_failures.extend({"candidate": str(candidate_id), **failure} for failure in failures)
    winner = min(scores, key=lambda score: (float(score["validation_mae"]), int(score["candidate"])))
    return templates.iloc[[int(winner["candidate"])]].copy(), scores, selection_failures


def _template_record(template: pd.DataFrame) -> dict[str, object]:
    row = template.iloc[0].dropna().to_dict()
    return {str(key): value.item() if hasattr(value, "item") else value for key, value in row.items()}


def run_experiment(data: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    series, audit = build_monthly_series(data)
    started = time.perf_counter()
    templates = search_templates(series.loc[:TRAIN_END], output)
    search_seconds = time.perf_counter() - started
    selection_started = time.perf_counter()
    selected, scores, selection_failures = select_template(series, templates)
    selection_seconds = time.perf_counter() - selection_started
    selected.to_csv(output / "autots_selected_template.csv", index=False)
    target_months = series.index[series.index > TRAIN_END]
    forecast_started = time.perf_counter()
    forecast_failures: list[dict[str, str]] = []
    prediction = expanding_forecast(series, target_months, selected, forecast_failures)
    forecast_seconds = time.perf_counter() - forecast_started
    frame = pd.DataFrame({"month": target_months, "actual": series.loc[target_months], "forecast": prediction, "residual": series.loc[target_months] - prediction, "model": "autots", "split": [split_label(month) for month in target_months]})
    frame.to_csv(output / "autots_forecasts.csv", index=False)
    metrics = []
    for split in ("validation", "test"):
        subset = frame[frame.split == split].set_index("month")
        metrics.append({"model": "autots", "split": split, **_metrics(subset.actual, subset.forecast, series.loc[:TRAIN_END])})
    pd.DataFrame(metrics).to_csv(output / "autots_metrics.csv", index=False)
    # metrics.csv is the experiment-wide comparison table used by existing
    # tooling. Preserve its classical rows while making AutoTS discoverable.
    combined_metrics_path = output / "metrics.csv"
    existing_metrics = pd.read_csv(combined_metrics_path) if combined_metrics_path.exists() else pd.DataFrame()
    existing_metrics = existing_metrics[existing_metrics.model != "autots"] if "model" in existing_metrics else existing_metrics
    pd.concat([existing_metrics, pd.DataFrame(metrics)], ignore_index=True).to_csv(combined_metrics_path, index=False)
    report = {
        "model": "autots", "forecast_months": len(frame), "validation_months": int((frame.split == "validation").sum()), "test_months": int((frame.split == "test").sum()),
        "search_options": SEARCH_OPTIONS, "candidate_scores": scores, "selected_template": _template_record(selected),
        "rounding": "nonnegative nearest whole count, half-up", "fallback": "seasonal naive when an imported template fails or predicts missing/nonfinite output",
        "runtime_seconds": {"search": search_seconds, "validation_selection": selection_seconds, "expanding_forecast": forecast_seconds, "total": time.perf_counter() - started},
        "failures": {"validation": selection_failures, "forecast": forecast_failures}, "retained_sids": int(audit.SID.nunique()),
    }
    (output / "autots_audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    _svg(output / "autots_forecast_vs_actual.svg", "AutoTS monthly forecast vs actual", [(frame.actual.tolist(), "#111111"), (frame.forecast.tolist(), "#4e79a7")])
    _svg(output / "autots_model_comparison.svg", "AutoTS MAE: validation vs test", [], [float(row["mae"]) for row in metrics])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/monthly_activity_ts"))
    args = parser.parse_args()
    print(json.dumps(run_experiment(load_ibtracs(args.csv), args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
