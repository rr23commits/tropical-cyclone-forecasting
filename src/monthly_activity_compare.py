"""Unified comparison of every monthly forecasting approach.

Reads the archived one-step forecasts of all models (classical, ML, deep
learning, volatility, multivariate, Transformer, ensemble) and recomputes one
common metric set - RMSE, MAE, MAPE (months with storms), sMAPE, MASE, WAPE and
bias - on the shared validation/test months.  Also writes:

* ``comparison_metrics.csv``           all models x {validation, test}
* ``comparison_table.md``              test table sorted by RMSE (+ validation)
* ``comparison_by_season.csv``         MAE/RMSE per hurricane-season phase
* ``comparison_dm_tests.csv``          Diebold-Mariano tests vs. seasonal naive
* ``comparison_tuning_gain.csv``       baseline vs. fine-tuned models
* ``comparison_*.png``                 bar charts, overlay, heatmap, tuning gain

Nothing here refits a model; it is a read-only evaluation of archived forecasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .monthly_activity import TRAIN_END
from .monthly_activity_common import (
    DEFAULT_RESULTS_DIR,
    diebold_mariano,
    extended_metrics,
    load_monthly_series,
    season_of,
)

# model id -> (label, family, forecast file, column filter)
MODELS: dict[str, tuple[str, str, str]] = {
    "seasonal_naive": ("Seasonal naive", "Baseline", "forecasts.csv"),
    "arima": ("ARIMA", "Classical", "forecasts.csv"),
    "sarima": ("SARIMA", "Classical", "forecasts.csv"),
    "holt_winters": ("Holt-Winters", "Classical", "forecasts.csv"),
    "garch": ("ARX-GARCH", "Volatility", "garch_forecasts.csv"),
    "var": ("VARX (half-month)", "Multivariate", "var_forecasts.csv"),
    "prophet": ("Prophet", "Additive/Bayesian", "prophet_forecasts.csv"),
    "prophet_tuned": ("Prophet (tuned)", "Additive/Bayesian", "prophet_tuned_forecasts.csv"),
    "xgboost": ("XGBoost", "Tree ensemble", "xgboost_forecasts.csv"),
    "xgboost_tuned": ("XGBoost (tuned)", "Tree ensemble", "xgboost_tuned_forecasts.csv"),
    "lstm": ("LSTM", "Recurrent NN", "lstm_forecasts.csv"),
    "lstm_tuned": ("LSTM (tuned)", "Recurrent NN", "lstm_tuned_forecasts.csv"),
    "gru": ("GRU", "Recurrent NN", "gru_forecasts.csv"),
    "gru_tuned": ("GRU (tuned)", "Recurrent NN", "gru_tuned_forecasts.csv"),
    "patchtst": ("PatchTST", "Transformer", "patchtst_forecasts.csv"),
    "autots": ("AutoTS", "AutoML", "autots_forecasts.csv"),
    "ensemble_xgboost_prophet": ("XGBoost+Prophet ensemble", "Ensemble", "ensemble_forecasts.csv"),
}
TUNING_PAIRS = [("xgboost", "xgboost_tuned"), ("prophet", "prophet_tuned"), ("lstm", "lstm_tuned"), ("gru", "gru_tuned")]
COLORS = {"Baseline": "#9c9c9c", "Classical": "#4e79a7", "Volatility": "#59a14f", "Multivariate": "#edc948", "Additive/Bayesian": "#76b7b2",
          "Tree ensemble": "#f28e2b", "Recurrent NN": "#e15759", "Transformer": "#b07aa1", "AutoML": "#bab0ac", "Ensemble": "#ff9da7"}


def load_all(output: Path) -> pd.DataFrame:
    """Long frame: month, split, actual, forecast, model - one row per model per month."""
    frames = []
    for model, (_, _, filename) in MODELS.items():
        path = output / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "model" in frame:
            frame = frame[frame["model"] == model]
        if frame.empty:
            continue
        frames.append(frame[["month", "split", "actual", "forecast"]].assign(model=model))
    data = pd.concat(frames, ignore_index=True)
    data["month"] = pd.to_datetime(data["month"], utc=True)
    reference = data[data.model == "seasonal_naive"][["month", "actual"]].reset_index(drop=True)
    for model, part in data.groupby("model"):
        part = part.sort_values("month").reset_index(drop=True)
        if len(part) != len(reference) or not np.allclose(part.actual, reference.actual) or not (part.month == reference.month).all():
            raise ValueError(f"{model} forecasts are not on the shared monthly index/actuals")
    return data


def metric_table(data: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    rows = []
    train = series.loc[:TRAIN_END]
    for (model, split), part in data.groupby(["model", "split"]):
        label, family, _ = MODELS[model]
        rows.append({"model": model, "label": label, "family": family, "split": split, **extended_metrics(part.actual, part.forecast, train)})
    return pd.DataFrame(rows)


def season_table(data: pd.DataFrame) -> pd.DataFrame:
    test = data[data.split == "test"].copy()
    test["season_phase"] = [season_of(m.month) for m in test.month]
    rows = []
    for (model, phase), part in test.groupby(["model", "season_phase"]):
        err = part.forecast - part.actual
        rows.append({"model": model, "season_phase": phase, "n": len(part), "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean())})
    return pd.DataFrame(rows)


def dm_table(data: pd.DataFrame, best_model: str) -> pd.DataFrame:
    test = data[data.split == "test"].pivot(index="month", columns="model", values="forecast")
    actual = data[(data.split == "test") & (data.model == "seasonal_naive")].set_index("month").actual.reindex(test.index).to_numpy()
    rows = []
    for model in test.columns:
        for reference in ("seasonal_naive", best_model):
            if model == reference:
                continue
            stat, p = diebold_mariano(actual, test[model].to_numpy(), test[reference].to_numpy(), power=2)
            rows.append({"model": model, "reference": reference, "loss": "squared error", "dm_statistic": stat, "p_value": p, "model_better": bool(stat < 0)})
    return pd.DataFrame(rows)


def markdown_table(metrics: pd.DataFrame) -> str:
    def block(split: str) -> str:
        part = metrics[metrics.split == split].sort_values("rmse")
        lines = ["| Rank | Model | Family | RMSE | MAE | MAPE % (y>0) | sMAPE % | MASE | WAPE % | Bias |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for rank, row in enumerate(part.itertuples(), 1):
            lines.append(f"| {rank} | {row.label} | {row.family} | {row.rmse:.3f} | {row.mae:.3f} | {row.mape:.1f} | {row.smape:.1f} | {row.mase:.3f} | {row.wape:.1f} | {row.bias:+.3f} |")
        return "\n".join(lines)

    n_test = int(metrics[metrics.split == "test"].n.iloc[0])
    positive = int(metrics[metrics.split == "test"].mape_n.iloc[0])
    return (f"### Held-out test (2020-01..2025-12, n={n_test}; MAPE uses the {positive} months with >=1 storm), sorted by RMSE\n\n{block('test')}\n\n"
            f"### Validation (2016-01..2019-12), sorted by RMSE\n\n{block('validation')}\n")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plots(output: Path, data: pd.DataFrame, metrics: pd.DataFrame, seasons: pd.DataFrame) -> None:
    plt = _plt()
    test = metrics[metrics.split == "test"].sort_values("rmse")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, metric, title in zip(axes, ("rmse", "mae", "mape"), ("RMSE", "MAE", "MAPE % (months with storms)")):
        ordered = test.sort_values("rmse", ascending=False)  # same row order in every panel (shared y-axis)
        ax.barh(ordered.label, ordered[metric], color=[COLORS[f] for f in ordered.family])
        ax.set_title(f"Test {title} (lower is better)", fontsize=10)
        for y, v in enumerate(ordered[metric]):
            ax.text(v, y, f" {v:.2f}" if metric != "mape" else f" {v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "comparison_error_bars.png", dpi=130)
    plt.close(fig)

    top = list(test.head(4).model)
    show = list(dict.fromkeys(top + ["seasonal_naive"]))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    reference = data[(data.model == "seasonal_naive") & (data.split == "test")].sort_values("month")
    months = reference.month.dt.tz_localize(None)
    ax.plot(months, reference.actual, color="black", lw=2, label="actual")
    for model in show:
        part = data[(data.model == model) & (data.split == "test")].sort_values("month")
        ax.plot(months, part.forecast, lw=1.1, alpha=0.85, label=MODELS[model][0], ls="--" if model == "seasonal_naive" else "-")
    ax.set_title("Test period: actual storm counts vs. the four best models (by RMSE) and seasonal naive")
    ax.set_ylabel("storms / month")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "comparison_test_overlay.png", dpi=130)
    plt.close(fig)

    pivot = seasons.pivot(index="model", columns="season_phase", values="mae")
    pivot = pivot.loc[[m for m in test.model if m in pivot.index]]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), [MODELS[m][0] for m in pivot.index], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, label="test MAE")
    ax.set_title("Where models err: test MAE by season phase")
    fig.tight_layout()
    fig.savefig(output / "comparison_error_by_season.png", dpi=130)
    plt.close(fig)

    pairs = [(a, b) for a, b in TUNING_PAIRS if a in set(metrics.model) and b in set(metrics.model)]
    if pairs:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        width = 0.38
        for i, (base, tuned) in enumerate(pairs):
            b = metrics[(metrics.model == base) & (metrics.split == "test")].rmse.iloc[0]
            t = metrics[(metrics.model == tuned) & (metrics.split == "test")].rmse.iloc[0]
            ax.bar(i - width / 2, b, width, color="#bab0ac", label="baseline" if i == 0 else None)
            ax.bar(i + width / 2, t, width, color="#4e79a7", label="fine-tuned" if i == 0 else None)
            ax.text(i + width / 2, t, f"{(t - b) / b * 100:+.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(pairs)), [MODELS[b][0] for b, _ in pairs])
        ax.set_ylabel("test RMSE")
        ax.set_title("Effect of fine-tuning (validation-selected) on held-out RMSE")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / "comparison_tuning_gain.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    val = metrics[metrics.split == "validation"].set_index("model").rmse
    tst = metrics[metrics.split == "test"].set_index("model")
    for model, row in tst.iterrows():
        ax.scatter(val[model], row.rmse, color=COLORS[row.family], s=55)
        ax.annotate(row.label, (val[model], row.rmse), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [min(val.min(), tst.rmse.min()) - 0.05, max(val.max(), tst.rmse.max()) + 0.05]
    ax.plot(lim, lim, color="#999999", ls=":")
    ax.set_xlabel("validation RMSE (used for selection)")
    ax.set_ylabel("test RMSE (held out)")
    ax.set_title("Selection vs. held-out RMSE (above the line = worse on test)")
    fig.tight_layout()
    fig.savefig(output / "comparison_validation_vs_test.png", dpi=130)
    plt.close(fig)


def run(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    series = load_monthly_series(output)
    data = load_all(output)
    metrics = metric_table(data, series)
    metrics.to_csv(output / "comparison_metrics.csv", index=False)
    seasons = season_table(data)
    seasons.to_csv(output / "comparison_by_season.csv", index=False)
    test = metrics[metrics.split == "test"].sort_values("rmse")
    best = str(test.model.iloc[0])
    dm_table(data, best).to_csv(output / "comparison_dm_tests.csv", index=False)
    gains = []
    for base, tuned in TUNING_PAIRS:
        if base in set(metrics.model) and tuned in set(metrics.model):
            for split in ("validation", "test"):
                b = metrics[(metrics.model == base) & (metrics.split == split)].iloc[0]
                t = metrics[(metrics.model == tuned) & (metrics.split == split)].iloc[0]
                gains.append({"model": MODELS[base][0], "split": split, **{f"{m}_baseline": b[m] for m in ("rmse", "mae", "mape")}, **{f"{m}_tuned": t[m] for m in ("rmse", "mae", "mape")},
                              "rmse_change_pct": (t.rmse - b.rmse) / b.rmse * 100, "mae_change_pct": (t.mae - b.mae) / b.mae * 100})
    pd.DataFrame(gains).to_csv(output / "comparison_tuning_gain.csv", index=False)
    (output / "comparison_table.md").write_text(markdown_table(metrics))
    plots(output, data, metrics, seasons)
    return {"models": sorted(set(metrics.model)), "best_test_rmse": best, "best_test_rmse_value": float(test.rmse.iloc[0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
