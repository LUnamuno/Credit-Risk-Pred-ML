"""Probability calibration: metrics and calibrated-model construction.

Calibration is learned on the training split only (internal cross-validation), so the
validation split stays untouched for choosing between calibration methods and for choosing the
decision threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# "none" keeps the raw model probabilities; the others wrap the model in cross-validated
# calibration (sigmoid = Platt scaling, isotonic = monotonic non-parametric mapping).
CALIBRATION_METHODS: tuple[str, ...] = ("none", "sigmoid", "isotonic")
CALIBRATION_FOLDS = 5


def expected_calibration_error(y_true: Any, y_proba: Any, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins.

    Weighted average of ``|observed default rate - mean predicted probability|`` over bins.
    0 means perfectly calibrated; it is a summary, so read it together with the Brier score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    bin_ids = np.minimum((y_proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_proba[mask].mean())
    return float(ece)


def build_calibrated(pipeline: BaseEstimator, method: str, seed: int) -> BaseEstimator:
    """Return ``pipeline`` unchanged (``none``) or wrapped in cross-validated calibration.

    The returned object is unfitted and exposes ``fit`` / ``predict_proba`` like a Pipeline.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration '{method}'. Available: {list(CALIBRATION_METHODS)}")
    if method == "none":
        return pipeline
    cv = StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(pipeline, method=method, cv=cv)
