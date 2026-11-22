"""Vector autoregression (VAR) for monthly storm activity.

A VAR needs more than one series.  The frozen IBTrACS extract only yields storm
counts, so the default system is a two-variable decomposition of the count
that uses nothing but the retained-storm audit:

    first_half_t  = storms first observed on days 1-15 of month t
    second_half_t = storms first observed on days 16-end of month t
    count_t       = first_half_t + second_half_t        (the forecast target)

The VAR lets each half-month respond to the lagged state of both, i.e. it
captures storm *clustering* across consecutive weeks.  Deterministic seasonality
enters as exogenous harmonics (VARX).

Real climate drivers (ENSO/ONI, AMO, MDR SST, ...) can be added with
``--exog path.csv`` (columns: ``month`` plus numeric columns); they become
extra endogenous variables, and Granger causality towards the count is reported.
No such driver data ships with this repository, so none is used by default.

Protocol: expanding-window one-step-ahead; lag order chosen on validation MAE.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from .monthly_activity import TRAIN_END
from .monthly_activity_common import (
    DEFAULT_RESULTS_DIR,
    forecast_months,
    load_monthly_series,
    png_forecast_plot,
    split_frame,
    validation_months,
    write_model_outputs,
)
from .monthly_activity_garch import harmonics

LAG_GRID = (1, 2, 3, 6, 12)
HARMONICS = 3


def half_month_system(series: pd.Series, audit: pd.DataFrame) -> pd.DataFrame:
    """Return the monthly [first_half, second_half] system aligned to ``series``."""
    first = pd.to_datetime(audit["ISO_TIME"], utc=True)
    late = (first.dt.day > 15).astype(int)
    frame = pd.DataFrame({"month": pd.to_datetime(audit["month"], utc=True), "late": late})
    counts = frame.groupby(["month", "late"]).size().unstack(fill_value=0).reindex(series.index, fill_value=0)
    counts = counts.reindex(columns=[0, 1], fill_value=0).astype(float)
    counts.columns = ["first_half", "second_half"]
    if not np.allclose(counts.sum(axis=1).to_numpy(), series.to_numpy()):
        raise ValueError("half-month split does not sum to the monthly count series")
    return counts


def load_system(output: Path, series: pd.Series, exog_path: Path | None) -> pd.DataFrame:
    if exog_path is not None:
        extra = pd.read_csv(exog_path)
        extra.index = pd.DatetimeIndex(pd.to_datetime(extra.pop("month"), utc=True))
        system = pd.concat([series.rename("count"), extra.reindex(series.index)], axis=1)
        if system.isna().any().any():
            raise ValueError("exogenous driver file must cover every month 1980-01..2025-12")
        return system
    audit = pd.read_csv(output / "retained_storm_audit.csv")
    return half_month_system(series, audit)


def forecast_one(system: pd.DataFrame, target_month: pd.Timestamp, lags: int, count_is_sum: bool) -> float:
    pos = system.index.get_loc(target_month)
    history = system.iloc[:pos]
    exog = harmonics(history.index, HARMONICS)
    exog_target = harmonics(pd.DatetimeIndex([target_month]), HARMONICS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = VAR(history.to_numpy(), exog=exog).fit(lags, trend="c")
        step = fit.forecast(history.to_numpy()[-lags:], steps=1, exog_future=exog_target)[0]
    return float(step.sum() if count_is_sum else step[0])


def expanding(system: pd.DataFrame, months: pd.DatetimeIndex, lags: int, count_is_sum: bool) -> pd.Series:
    return pd.Series([forecast_one(system, m, lags, count_is_sum) for m in months], index=months, name="forecast")


def run_experiment(output_dir: str | Path, exog_path: Path | None = None) -> dict[str, object]:
    output = Path(output_dir)
    series = load_monthly_series(output)
    system = load_system(output, series, exog_path)
    count_is_sum = exog_path is None
    val = validation_months(series)
    trials = []
    for lags in LAG_GRID:
        pred = expanding(system, val, lags, count_is_sum).to_numpy()
        actual = series.loc[val].to_numpy()
        trials.append({"lags": lags, "validation_mae": float(np.mean(np.abs(pred - actual))), "validation_rmse": float(np.sqrt(np.mean((pred - actual) ** 2)))})
        print(f"  lags={lags} val MAE={trials[-1]['validation_mae']:.3f}", flush=True)
    pd.DataFrame(trials).to_csv(output / "var_search.csv", index=False)
    best = min(trials, key=lambda r: (r["validation_mae"], r["validation_rmse"]))
    lags = int(best["lags"])
    frame = split_frame(series, expanding(system, forecast_months(series), lags, count_is_sum), "var")

    # Granger causality on the training window with the selected lag order.
    train = system.loc[:TRAIN_END]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = VAR(train.to_numpy(), exog=harmonics(train.index, HARMONICS)).fit(lags, trend="c")
    granger = {}
    names = list(system.columns)
    for i, caused in enumerate(names):
        for j, causing in enumerate(names):
            if i == j:
                continue
            test = fit.test_causality(i, [j], kind="f")
            granger[f"{causing} -> {caused}"] = {"f_stat": float(test.test_statistic), "p_value": float(test.pvalue)}
    audit = {"system_variables": names, "selected_lags": lags, "search_trials": len(trials), "selection_metric": "validation MAE",
             "exogenous_deterministic": f"{HARMONICS} seasonal harmonics", "external_drivers_file": str(exog_path) if exog_path else None,
             "granger_causality_train": granger}
    rows = write_model_outputs(output, series, frame, "var", audit)
    png_forecast_plot(output / "var_forecast_vs_actual.png", frame, "VAR one-step forecast vs actual", "#edc948")
    return {"model": "var", "selected_lags": lags, "metrics": rows, "granger": granger}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--exog", type=Path, default=None, help="optional monthly driver CSV (month + numeric columns)")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output_dir, args.exog), indent=2, default=str))


if __name__ == "__main__":
    main()
