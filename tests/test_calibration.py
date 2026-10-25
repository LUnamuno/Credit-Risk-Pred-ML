from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from credit_risk.calibration import (
    CALIBRATION_METHODS,
    build_calibrated,
    expected_calibration_error,
)
from credit_risk.train import build_pipeline


def test_ece_is_zero_for_perfectly_calibrated_bins():
    # Bin [0.2, 0.3): predicted 0.25 on 4 rows, 1 of 4 is positive -> observed 0.25.
    y = np.array([1, 0, 0, 0])
    p = np.full(4, 0.25)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    y = np.array([0, 0, 0, 0])
    p = np.full(4, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.9)


def test_ece_handles_probability_of_exactly_one():
    assert expected_calibration_error(np.array([1, 1]), np.array([1.0, 1.0])) == pytest.approx(0.0)


def test_build_calibrated_none_returns_same_pipeline():
    pipeline = build_pipeline()
    assert build_calibrated(pipeline, "none", seed=0) is pipeline


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_build_calibrated_wraps_pipeline(method):
    calibrated = build_calibrated(build_pipeline(), method, seed=0)
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert calibrated.method == method


def test_build_calibrated_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown calibration"):
        build_calibrated(build_pipeline(), "nope", seed=0)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_every_method_yields_valid_probabilities(method, synthetic_df):
    from credit_risk.data import FEATURES, TARGET

    model = build_calibrated(build_pipeline(), method, seed=0)
    model.fit(synthetic_df[FEATURES], synthetic_df[TARGET])
    proba = model.predict_proba(synthetic_df[FEATURES])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
