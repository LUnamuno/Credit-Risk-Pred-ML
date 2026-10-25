from __future__ import annotations

import numpy as np
import pytest

from credit_risk.decision import (
    APPROVE,
    REJECT,
    CostAssumptions,
    DecisionPolicy,
    baseline_costs,
    confusion_counts,
    evaluate_decision,
    expected_cost,
    find_best_threshold,
)

COSTS = CostAssumptions(cost_false_negative=5.0, cost_false_positive=1.0)


def test_theoretical_threshold_is_cost_ratio():
    assert COSTS.theoretical_threshold == pytest.approx(1 / 6)


def test_costs_must_be_positive():
    with pytest.raises(ValueError):
        CostAssumptions(cost_false_negative=0.0)
    with pytest.raises(ValueError):
        CostAssumptions(cost_false_positive=-1.0)


def test_confusion_counts_and_cost_by_hand():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.05])
    # threshold 0.5 -> rejects rows 0 and 2: tp=1, fp=1, fn=1 (row 1), tn=2
    assert confusion_counts(y, p, 0.5) == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    assert expected_cost(y, p, 0.5, COSTS) == pytest.approx((5 * 1 + 1 * 1) / 5)


def test_threshold_is_inclusive_reject():
    assert confusion_counts(np.array([1]), np.array([0.3]), 0.3)["tp"] == 1


def test_best_threshold_close_to_theoretical_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 100_000)
    y = rng.uniform(0, 1, p.size) < p  # perfectly calibrated by construction
    best = find_best_threshold(y, p, COSTS)
    assert best == pytest.approx(COSTS.theoretical_threshold, abs=0.03)


def test_best_threshold_is_never_worse_than_default_threshold():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5_000)
    y = rng.uniform(0, 1, p.size) < p
    best = find_best_threshold(y, p, COSTS)
    assert expected_cost(y, p, best, COSTS) <= expected_cost(y, p, 0.5, COSTS)


def test_evaluate_decision_rates():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1])
    result = evaluate_decision(y, p, 0.5, COSTS)
    assert result["approval_rate"] == pytest.approx(0.5)
    assert result["default_capture_rate"] == pytest.approx(0.5)
    assert result["rejection_precision"] == pytest.approx(0.5)


def test_evaluate_decision_with_no_rejections_does_not_divide_by_zero():
    result = evaluate_decision(np.array([0, 1]), np.array([0.1, 0.2]), 0.9, COSTS)
    assert result["rejection_precision"] == 0.0
    assert result["approval_rate"] == 1.0


def test_baselines_approve_all_and_reject_all():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    base = baseline_costs(y, p, COSTS)
    assert base["approve_everyone"] == pytest.approx(5 * 1 / 4)
    assert base["reject_everyone"] == pytest.approx(1 * 3 / 4)


def test_policy_decide_and_roundtrip(tmp_path):
    policy = DecisionPolicy(threshold=0.2, costs=COSTS)
    assert list(policy.decide([0.1, 0.2, 0.9])) == [APPROVE, REJECT, REJECT]
    policy.save(tmp_path / "sub" / "decision.json")
    assert DecisionPolicy.load(tmp_path / "sub" / "decision.json") == policy


def test_policy_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DecisionPolicy.load(tmp_path / "nope.json")
