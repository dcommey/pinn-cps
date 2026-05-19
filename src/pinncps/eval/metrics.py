"""Detection metrics: precision, recall, F1, ROC/PR AUC, detection delay."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _flatten(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return scores.reshape(-1), labels.reshape(-1)


def threshold_from_validation(
    nominal_scores: np.ndarray,
    quantile: float = 0.99,
) -> float:
    """Set the threshold at a high quantile of nominal scores."""
    return float(np.quantile(nominal_scores.reshape(-1), quantile))


def compute_detection_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Point-level (per-step) detection metrics across all timesteps."""
    s_flat, y_flat = _flatten(scores, labels)
    pred = (s_flat >= threshold).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(
        y_flat, pred, average="binary", zero_division=0,
    )
    # AUC requires both classes present.
    try:
        roc = float(roc_auc_score(y_flat, s_flat))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_flat, s_flat))
    except ValueError:
        pr = float("nan")
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc,
        "pr_auc": pr,
    }


def detection_delay(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Average number of steps from attack onset to first detection.

    For each trajectory we identify the first attacked step ``t_a`` and the
    first step ``t_d >= t_a`` where ``score >= threshold``.  ``delay = t_d - t_a``.
    If no detection occurs we record the trajectory horizon as a censored value.

    Reports mean delay (excluding misses) and the miss fraction.
    """
    if scores.ndim == 1:
        scores = scores[None]; labels = labels[None]
    N, T = scores.shape
    delays = []
    misses = 0
    censored_delays = []
    for i in range(N):
        atk = np.where(labels[i] > 0)[0]
        if atk.size == 0:
            continue
        t_a = int(atk[0])
        det = np.where(scores[i, t_a:] >= threshold)[0]
        if det.size == 0:
            misses += 1
            censored_delays.append(T - t_a)
            continue
        d = int(det[0])
        delays.append(d)
        censored_delays.append(d)
    if not censored_delays:
        return {"mean_delay": float("nan"), "median_delay": float("nan"), "miss_rate": float("nan")}
    return {
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "miss_rate": float(misses) / float(len(censored_delays)),
        "mean_delay_with_censoring": float(np.mean(censored_delays)),
    }
