"""
scripts/generate_plots.py — Produce all submission visualisations.

Reads:
  data/models/flexibility/metrics.json
  data/models/flexibility/bias_report.json
  data/processed/flexibility_features.parquet  (for target distributions)

Writes to data/models/flexibility/plots/:
  01_horizon_rmse_comparison.png
  02_model_selection.png
  03_shap_top10.png
  04_bias_sex.png
  05_bias_age.png
  06_hyperparam_sensitivity.png
  07_score_distribution.png

Usage:
  python scripts/generate_plots.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

AIRFLOW_HOME  = os.getenv("AIRFLOW_HOME", "/opt/airflow")
MODELS_DIR    = Path(f"{AIRFLOW_HOME}/data/models/flexibility")
PLOTS_DIR     = MODELS_DIR / "plots"
FEATURES_PATH = Path(f"{AIRFLOW_HOME}/data/processed/flexibility_features.parquet")

HORIZONS       = [1, 3, 7, 14]
HORIZON_LABELS = [f"+{h}d" for h in HORIZONS]

PALETTE = {
    "primary":   "#1D9E75",
    "secondary": "#7F77DD",
    "accent":    "#D85A30",
    "neutral":   "#888780",
    "warn":      "#BA7517",
    "grid":      "#E8E8E4",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})


def _save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_horizon_rmse(metrics):
    model_results = metrics.get("model_comparison", {})
    if not model_results:
        model_results = {"XGBoost (best)": metrics["test_metrics"]}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x      = np.arange(len(HORIZONS))
    n      = len(model_results)
    width  = 0.7 / n
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["neutral"]]

    for i, (name, result) in enumerate(model_results.items()):
        rmses  = [result.get(f"d{h}", {}).get("rmse", 0) for h in HORIZONS]
        offset = (i - n / 2 + 0.5) * width
        bars   = ax.bar(x + offset, rmses, width, label=name,
                        color=colors[i % len(colors)], alpha=0.88, zorder=3)
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE (score points)")
    ax.set_title("Model comparison — RMSE per forecast horizon")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    _save(fig, "01_horizon_rmse_comparison.png")


def plot_model_selection(metrics):
    model_results = metrics.get("model_comparison", {})
    if not model_results:
        model_results = {"XGBoost": metrics["test_metrics"]}

    names    = list(model_results.keys())
    rmses    = [model_results[m].get("d7", {}).get("rmse", 0) for m in names]
    r2s      = [model_results[m].get("d7", {}).get("r2",   0) for m in names]
    best_idx = int(np.argmin(rmses))
    bar_colors = [PALETTE["primary"] if i == best_idx else PALETTE["neutral"]
                  for i in range(len(names))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = ax1.bar(names, rmses, color=bar_colors, alpha=0.88, zorder=3)
    for bar, val in zip(bars1, rmses):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Test RMSE at +7d horizon")
    ax1.set_ylabel("RMSE (lower = better)")
    ax1.set_ylim(0, max(rmses) * 1.2 if rmses else 1)

    bars2 = ax2.bar(names, r2s, color=bar_colors, alpha=0.88, zorder=3)
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Test R² at +7d horizon")
    ax2.set_ylabel("R² (higher = better)")

    patch = mpatches.Patch(color=PALETTE["primary"], label=f"Selected: {names[best_idx]}")
    fig.legend(handles=[patch], loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Model selection — +7d forecast horizon", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_model_selection.png")


def plot_shap(metrics):
    shap_data = metrics.get("shap_top10", {})
    if not shap_data:
        print("  WARNING: no SHAP data in metrics.json — skipping plot 03")
        return

    features = list(shap_data.keys())[:10]
    values   = [shap_data[f] for f in features]
    order    = np.argsort(values)
    features = [features[i] for i in order]
    values   = [values[i]   for i in order]
    colors   = [PALETTE["primary"] if v == max(values) else PALETTE["secondary"] for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, values, color=colors, alpha=0.88, zorder=3)
    for v, f in zip(values, features):
        ax.text(v + max(values) * 0.01, features.index(f), f"{v:.4f}",
                va="center", fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value| (impact on +7d prediction)")
    ax.set_title("Feature importance — SHAP (top 10, horizon +7d)")
    ax.set_xlim(0, max(values) * 1.25)
    _save(fig, "03_shap_top10.png")


def plot_bias_slice(bias, slice_col, plot_name, title):
    slice_data = bias.get("slices", {}).get(slice_col)
    if not slice_data:
        print(f"  WARNING: no bias data for '{slice_col}' — skipping {plot_name}")
        return

    overall  = slice_data["overall_rmse"]
    by_group = slice_data["by_group"]
    groups   = list(by_group.keys())
    rmses    = [by_group[g] for g in groups]
    flagged  = bias.get("flagged", [])

    bar_colors = []
    for g, r in zip(groups, rmses):
        if any(f"{slice_col}={g}" in flag for flag in flagged):
            bar_colors.append(PALETTE["accent"])
        elif r > overall * 1.2:
            bar_colors.append(PALETTE["warn"])
        else:
            bar_colors.append(PALETTE["primary"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(groups, rmses, color=bar_colors, alpha=0.88, zorder=3)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(overall, color=PALETTE["secondary"], linewidth=1.5,
               linestyle="--", zorder=4, label=f"Overall RMSE: {overall:.3f}")
    ax.set_ylabel("RMSE (score points, horizon +7d)")
    ax.set_title(title)
    ax.set_ylim(0, max(rmses) * 1.25 if rmses else 1)

    patches = [
        mpatches.Patch(color=PALETTE["primary"], label="Within 20% of overall"),
        mpatches.Patch(color=PALETTE["warn"],    label=">20% worse"),
        mpatches.Patch(color=PALETTE["accent"],  label="Fairlearn flagged (>50% worse)"),
    ]
    ax.legend(handles=patches, fontsize=8.5, framealpha=0.9)
    _save(fig, plot_name)


def plot_hyperparam_sensitivity(metrics):
    sens = metrics.get("hyperparam_sensitivity", {})
    corr = sens.get("correlations", {})
    if not corr:
        print("  WARNING: no sensitivity data — skipping plot 06")
        return

    params = list(corr.keys())
    values = [corr[p] for p in params]
    order  = np.argsort(np.abs(values))[::-1]
    params = [params[i] for i in order]
    values = [values[i] for i in order]
    colors = [PALETTE["primary"] if v > 0 else PALETTE["accent"] for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(params[::-1], values[::-1], color=colors[::-1], alpha=0.88, zorder=3)
    ax.axvline(0, color=PALETTE["neutral"], linewidth=0.8, zorder=5)
    for i, (p, v) in enumerate(zip(params[::-1], values[::-1])):
        xpos = v + (0.01 if v >= 0 else -0.01)
        ax.text(xpos, i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlabel("Pearson correlation with CV RMSE\n(positive = higher value -> better RMSE)")
    ax.set_title("Hyperparameter sensitivity")
    patches = [
        mpatches.Patch(color=PALETTE["primary"], label="Positive effect"),
        mpatches.Patch(color=PALETTE["accent"],  label="Negative effect"),
    ]
    ax.legend(handles=patches, fontsize=9)
    _save(fig, "06_hyperparam_sensitivity.png")


def plot_score_distributions():
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not available — skipping plot 07")
        return
    if not FEATURES_PATH.exists():
        print("  WARNING: feature file not found — skipping plot 07")
        return

    df     = pd.read_parquet(str(FEATURES_PATH))
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["warn"]]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)

    for ax, h, color in zip(axes, HORIZONS, colors):
        col  = f"target_d{h}"
        data = df[col].dropna()
        ax.hist(data, bins=30, color=color, alpha=0.82, edgecolor="white",
                linewidth=0.4, zorder=3)
        ax.axvline(data.mean(), color="black", linewidth=1.2,
                   linestyle="--", label=f"mean={data.mean():.1f}")
        ax.set_title(f"Target +{h}d")
        ax.set_xlabel("Score (0-100)")
        ax.legend(fontsize=8.5)
        if ax == axes[0]:
            ax.set_ylabel("Count")

    fig.suptitle("Distribution of forecast targets by horizon", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_score_distribution.png")


def generate_all(metrics_path=None, bias_path=None):
    metrics_path = Path(metrics_path) if metrics_path else MODELS_DIR / "metrics.json"
    bias_path    = Path(bias_path)    if bias_path    else MODELS_DIR / "bias_report.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.json not found at {metrics_path}\n"
            "Run model_train.py first."
        )

    metrics = json.loads(metrics_path.read_text())
    bias    = json.loads(bias_path.read_text()) if bias_path.exists() else {}

    print(f"Generating plots -> {PLOTS_DIR}/")
    plot_horizon_rmse(metrics)
    plot_model_selection(metrics)
    plot_shap(metrics)
    plot_bias_slice(bias, "sex",        "04_bias_sex.png", "Bias by sex — RMSE at +7d horizon")
    plot_bias_slice(bias, "age_bucket", "05_bias_age.png", "Bias by age group — RMSE at +7d horizon")
    plot_hyperparam_sensitivity(metrics)
    plot_score_distributions()
    print(f"Done. {len(list(PLOTS_DIR.glob('*.png')))} plots in {PLOTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--bias",    default=None)
    args = parser.parse_args()
    generate_all(args.metrics, args.bias)