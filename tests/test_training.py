from __future__ import annotations

import json

import numpy as np
import pytest

from credit_risk.calibration import CALIBRATION_METHODS
from credit_risk.data import FEATURES
from credit_risk.decision import DEFAULT_COSTS, CostAssumptions, DecisionPolicy
from credit_risk.predict import load_pipeline, predict_proba
from credit_risk.train import build_pipeline, compute_metrics, train


@pytest.fixture()
def trained(synthetic_df, tmp_path):
    data_path = tmp_path / "data.csv"
    synthetic_df.to_csv(data_path, index=False)
    report = train(
        data_path,
        model_path=tmp_path / "pipeline.joblib",
        metrics_path=tmp_path / "metrics.json",
        expected_rows=None,
    )
    return report, tmp_path


def test_train_saves_model_and_metrics(trained):
    report, tmp_path = trained
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == report
    assert set(report["validation"]) == {"roc_auc", "pr_auc", "brier"}
    assert 0.5 < report["validation"]["roc_auc"] <= 1.0
    assert (tmp_path / "pipeline.joblib").exists()


def test_train_calibrates_and_saves_decision_policy(trained):
    report, tmp_path = trained
    assert report["calibration"]["selected"] in CALIBRATION_METHODS
    assert set(report["calibration"]["validation"]) == set(CALIBRATION_METHODS)
    # The selected method has the lowest validation Brier score.
    briers = {m: v["brier"] for m, v in report["calibration"]["validation"].items()}
    assert report["calibration"]["selected"] == min(briers, key=briers.get)

    policy = DecisionPolicy.load(tmp_path / "decision.json")
    assert policy.threshold == report["decision"]["threshold"]
    assert policy.costs == DEFAULT_COSTS
    assert 0 < policy.threshold < 1


def test_train_reports_test_split_evaluation(trained):
    report, _ = trained
    assert set(report["test"]) == {"roc_auc", "pr_auc", "brier", "ece"}
    assert report["decision"]["test"]["threshold"] == report["decision"]["threshold"]
    assert set(report["decision"]["test_baselines_cost_per_applicant"]) == {
        "approve_everyone",
        "reject_everyone",
        "threshold_0.5",
    }
    counts = report["decision"]["test"]
    assert counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] == report["n_test"]


def test_custom_costs_move_the_threshold(synthetic_df, tmp_path):
    data_path = tmp_path / "data.csv"
    synthetic_df.to_csv(data_path, index=False)
    kwargs = {"data_path": data_path, "expected_rows": None}
    cheap = train(
        model_path=tmp_path / "a.joblib",
        metrics_path=tmp_path / "a.json",
        costs=CostAssumptions(cost_false_negative=1.5, cost_false_positive=1.0),
        **kwargs,
    )
    costly = train(
        model_path=tmp_path / "b.joblib",
        metrics_path=tmp_path / "b.json",
        costs=CostAssumptions(cost_false_negative=20.0, cost_false_positive=1.0),
        **kwargs,
    )
    # Making defaults more expensive should never raise the threshold at which we reject.
    assert costly["decision"]["threshold"] <= cheap["decision"]["threshold"]


def test_training_is_reproducible(synthetic_df, tmp_path):
    data_path = tmp_path / "data.csv"
    synthetic_df.to_csv(data_path, index=False)
    kwargs = {"expected_rows": None, "data_path": data_path}
    r1 = train(model_path=tmp_path / "a.joblib", metrics_path=tmp_path / "a.json", **kwargs)
    r2 = train(model_path=tmp_path / "b.joblib", metrics_path=tmp_path / "b.json", **kwargs)
    assert r1 == r2


def test_predict_proba_in_unit_interval(trained, synthetic_df):
    _, tmp_path = trained
    pipeline = load_pipeline(tmp_path / "pipeline.joblib")
    proba = predict_proba(synthetic_df, pipeline)
    assert proba.shape == (len(synthetic_df),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_proba_accepts_single_record_with_unseen_category(trained, synthetic_df):
    _, tmp_path = trained
    pipeline = load_pipeline(tmp_path / "pipeline.joblib")
    record = synthetic_df[FEATURES].iloc[0].to_dict()
    record["EDUCATION"] = 99
    proba = predict_proba(record, pipeline)
    assert proba.shape == (1,)
    assert 0 <= proba[0] <= 1


def test_predict_proba_rejects_missing_columns(trained, synthetic_df):
    _, tmp_path = trained
    pipeline = load_pipeline(tmp_path / "pipeline.joblib")
    with pytest.raises(ValueError, match="Missing feature columns"):
        predict_proba(synthetic_df.drop(columns=["AGE"]), pipeline)


def test_load_pipeline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "nope.joblib")


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("nope")


def test_compute_metrics_perfect_ranking():
    m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["brier"] < 0.05
