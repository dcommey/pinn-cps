"""Publication-quality figure helpers.

Matplotlib is configured for IEEE-style figures: single-column width 3.5 in.,
serif font, 8 pt labels.  All figures are saved as both PDF (vector) and PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_trajectory_attack(
    clean_states: np.ndarray,
    attacked_obs: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str = "",
):
    _style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(clean_states[:, 0], clean_states[:, 1], "-", color="0.4", label="ground truth")
    ax.plot(attacked_obs[:, 0], attacked_obs[:, 1], "--", color="tab:red", label="observed")
    mask = labels > 0
    ax.scatter(attacked_obs[mask, 0], attacked_obs[mask, 1], s=4, color="tab:red", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    _save(fig, path)


def plot_roc_curves(
    scores_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]],
    path: Path,
    title: str = "",
):
    _style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for name, (scores, labels) in scores_by_method.items():
        fpr, tpr, _ = roc_curve(labels.reshape(-1), scores.reshape(-1))
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.6)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, path)


def plot_detection_delay(
    delays_by_method: Dict[str, np.ndarray],
    path: Path,
    title: str = "",
):
    _style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    methods = list(delays_by_method)
    data = [delays_by_method[m] for m in methods]
    ax.boxplot(data, tick_labels=methods, showfliers=False)
    ax.set_ylabel("detection delay [steps]")
    if title:
        ax.set_title(title)
    fig.autofmt_xdate(rotation=30)
    _save(fig, path)


def plot_residual_heatmap(
    residuals: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str = "",
):
    _style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(residuals, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("time step")
    ax.set_ylabel("trajectory")
    # Mark attacked regions with a thin red band.
    for i in range(labels.shape[0]):
        atk = np.where(labels[i] > 0)[0]
        if atk.size > 0:
            ax.hlines(i, atk[0], atk[-1], colors="white", linewidth=0.4)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    _save(fig, path)
