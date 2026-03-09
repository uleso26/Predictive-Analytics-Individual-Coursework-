"""
Metrics, calibration, and result-collection helpers.

All metric functions take (y_true, y_prob) or (y_true, y_pred) and return
a scalar.  `compute_all_metrics` returns a dict suitable for one row in
a results DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
)


# ── Individual metrics ────────────────────────────────────────────────

def safe_roc_auc(y_true, y_prob):
    """ROC-AUC; returns NaN if only one class present."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob):
    """PR-AUC (average precision); NaN-safe."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) with equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_acc - avg_conf)
    return ece / len(y_true)


# ── Composite metric dict ─────────────────────────────────────────────

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """Return dict of all project metrics.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_prob : array-like of predicted probabilities for class 1
    threshold : float, classification threshold for hard predictions
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


# ── Results collector ──────────────────────────────────────────────────

class ResultsCollector:
    """Accumulate per-model, per-horizon evaluation rows."""

    def __init__(self):
        self.rows = []

    def add(self, horizon: str, model_name: str, split: str,
            y_true, y_prob, threshold=0.5, extra: dict = None):
        """Compute metrics and store a results row."""
        metrics = compute_all_metrics(y_true, y_prob, threshold=threshold)
        row = {"horizon": horizon, "model": model_name, "split": split}
        row.update(metrics)
        if extra:
            row.update(extra)
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self, split="val") -> pd.DataFrame:
        """Return a pivot of ROC-AUC by horizon × model for the given split."""
        df = self.to_dataframe()
        df = df[df["split"] == split]
        if df.empty:
            return df
        return df.pivot_table(
            index="model", columns="horizon", values="roc_auc"
        ).round(4)
