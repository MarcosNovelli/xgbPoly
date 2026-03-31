"""
polybot/model/evaluate.py

Evaluation utilities for the calibrated XGB model.

Metrics
-------
- Brier score  : primary calibration metric (lower is better, baseline = 0.25)
- ROC AUC      : discrimination ability (0.50 = random, 0.60+ = strong)
- Log loss     : penalises confident wrong predictions
- Accuracy     : win rate at 0.5 threshold

Plots
-----
- Reliability diagram   : predicted probability vs actual frequency per bucket
- Brier score per fold  : stability across walk-forward folds
- ROC curve             : one line per fold
- Feature importance    : mean gain across folds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    log_loss,
    accuracy_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, p_green: np.ndarray) -> dict:
    """
    Compute evaluation metrics for one fold.

    Parameters
    ----------
    y_true  : binary labels (0/1)
    p_green : calibrated P(green) probabilities

    Returns
    -------
    dict with keys: brier, auc, logloss, accuracy, mean_edge, n_samples
    """
    brier     = brier_score_loss(y_true, p_green)
    auc       = roc_auc_score(y_true, p_green)
    ll        = log_loss(y_true, p_green)
    acc       = accuracy_score(y_true, (p_green >= 0.5).astype(int))
    mean_edge = float(np.mean(np.abs(p_green - 0.5)))

    return {
        "brier"     : round(brier, 5),
        "auc"       : round(auc, 5),
        "logloss"   : round(ll, 5),
        "accuracy"  : round(acc, 5),
        "mean_edge" : round(mean_edge, 5),
        "n_samples" : int(len(y_true)),
    }


def metrics_summary(fold_metrics: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-fold metrics into a summary DataFrame.

    Parameters
    ----------
    fold_metrics : list of dicts from compute_metrics(), one per fold

    Returns
    -------
    DataFrame with mean, std, min, max per metric
    """
    df = pd.DataFrame(fold_metrics)
    numeric = df.select_dtypes(include="number")
    summary = pd.concat([
        numeric.mean().rename("mean"),
        numeric.std().rename("std"),
        numeric.min().rename("min"),
        numeric.max().rename("max"),
    ], axis=1).T
    return summary


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true: np.ndarray,
    p_green: np.ndarray,
    n_bins: int = 10,
    ax: plt.Axes = None,
    title: str = "Reliability Diagram",
) -> plt.Axes:
    """
    Plot predicted probability vs actual frequency per bucket.

    A perfectly calibrated model lies on the diagonal.
    Points below = overconfident, points above = underconfident.
    Marker size reflects number of samples in each bucket.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, actual_freqs, counts = [], [], []

    for i in range(n_bins):
        mask = (p_green >= bins[i]) & (p_green < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        actual_freqs.append(y_true[mask].mean())
        counts.append(mask.sum())

    bin_centers  = np.array(bin_centers)
    actual_freqs = np.array(actual_freqs)
    counts       = np.array(counts)
    sizes        = 50 + 300 * (counts / counts.max())

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.scatter(bin_centers, actual_freqs, s=sizes, zorder=5, label="Model")
    ax.plot(bin_centers, actual_freqs, alpha=0.6)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    p_green: np.ndarray,
    ax: plt.Axes = None,
    label: str = "Model",
    color: str = "steelblue",
) -> plt.Axes:
    """Plot ROC curve for one fold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    fpr, tpr, _ = roc_curve(y_true, p_green)
    auc = roc_auc_score(y_true, p_green)

    ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_brier_per_fold(
    fold_metrics: list[dict],
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Bar chart of Brier score per fold.

    Bars above the 0.25 baseline (coin flip) are highlighted in red.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    briers = [m["brier"] for m in fold_metrics]
    folds  = [f"Fold {i+1}" for i in range(len(briers))]
    colors = ["tomato" if b >= 0.25 else "steelblue" for b in briers]

    ax.bar(folds, briers, color=colors, edgecolor="white")
    ax.axhline(0.25, color="black", linestyle="--", lw=1, label="Coin flip baseline (0.25)")
    ax.set_ylabel("Brier Score")
    ax.set_title("Brier Score per Walk-Forward Fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return ax


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Horizontal bar chart of top N features by mean gain across folds."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    idx = np.argsort(importances)[-top_n:]

    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Mean Gain")
    ax.set_title(f"Top {top_n} Feature Importances (mean across folds)")
    ax.grid(axis="x", alpha=0.3)

    return ax


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def plot_full_report(
    fold_results: list[dict],
    feature_names: list[str],
    output_path: Path,
) -> None:
    """
    Generate a single PNG with four panels:
      1. Reliability diagram  — pooled across all folds
      2. ROC curves           — one line per fold
      3. Brier score per fold — stability check
      4. Feature importance   — mean gain across folds

    Parameters
    ----------
    fold_results  : list of dicts, each with keys:
                    y_true, p_green, metrics, importances
    feature_names : list of feature column names
    output_path   : where to save the PNG
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])


    # 1. Reliability diagram — pool all fold predictions
    all_y = np.concatenate([r["y_true"] for r in fold_results])
    all_p = np.concatenate([r["p_green"] for r in fold_results])
    plot_reliability_diagram(all_y, all_p, ax=ax1, title="Reliability Diagram (all folds)")

    # 2. ROC curves — one per fold
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))
    for i, (r, c) in enumerate(zip(fold_results, colors)):
        plot_roc_curve(r["y_true"], r["p_green"], ax=ax2, label=f"Fold {i+1}", color=c)

    ax2.get_legend().remove()


    # 3. Brier per fold
    plot_brier_per_fold([r["metrics"] for r in fold_results], ax=ax3)

    # 4. Feature importance — mean across folds
    mean_importances = np.array([r["importances"] for r in fold_results]).mean(axis=0)
    plot_feature_importance(feature_names, mean_importances, ax=ax4)

    fig.suptitle("Walk-Forward Evaluation Report", fontsize=15, fontweight="bold")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved → {output_path}")