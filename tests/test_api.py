"""Integration tests: train on the synthetic fixture, then exercise the real FastAPI app."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from credit_risk.api import MAX_BATCH_SIZE, create_app  # noqa: E402
from credit_risk.data import FEATURES  # noqa: E402
from credit_risk.train import train  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(synthetic_df, tmp_path_factory):
    directory = tmp_path_factory.mktemp("model")
    data_path = directory / "data.csv"
    synthetic_df.to_csv(data_path, index=False)
    train(
        data_path,
        model_path=directory / "pipeline.joblib",
        metrics_path=directory / "metrics.json",
        expected_rows=None,
    )
    return directory


@pytest.fixture()
def client(model_dir):
    app = create_app(model_dir / "pipeline.joblib", model_dir / "decision.json")
    with TestClient(app) as test_client:  # the context manager runs the startup (model load)
        yield test_client


@pytest.fixture()
def applicant(synthetic_df):
    row = synthetic_df[FEATURES].iloc[0].to_dict()
    return {k: (int(v) if float(v).is_integer() else float(v)) for k, v in row.items()}


def test_health_reports_policy(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] < 1
    assert body["cost_false_negative"] > body["cost_false_positive"] > 0


def test_predict_returns_probability_and_consistent_decision(client, applicant):
    response = client.post("/predict", json=applicant)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_of_default"] <= 1
    expected = "reject" if body["probability_of_default"] >= body["threshold"] else "approve"
    assert body["decision"] == expected


def test_predict_matches_batch_for_same_applicant(client, applicant):
    single = client.post("/predict", json=applicant).json()
    batch = client.post("/predict/batch", json={"applicants": [applicant, applicant]}).json()
    assert len(batch["predictions"]) == 2
    for prediction in batch["predictions"]:
        # Floating-point sums can differ in the last digit depending on batch size (BLAS), so
        # compare the probability with a tolerance and everything else exactly.
        assert prediction["probability_of_default"] == pytest.approx(
            single["probability_of_default"], abs=1e-12
        )
        assert prediction["decision"] == single["decision"]
        assert prediction["threshold"] == single["threshold"]


def test_predict_accepts_undocumented_education_code(client, applicant):
    applicant["EDUCATION"] = 6
    assert client.post("/predict", json=applicant).status_code == 200


def test_rejects_missing_field(client, applicant):
    del applicant["AGE"]
    assert client.post("/predict", json=applicant).status_code == 422


def test_rejects_unknown_field(client, applicant):
    applicant["INCOME"] = 1000
    assert client.post("/predict", json=applicant).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [("SEX", 3), ("AGE", 5), ("LIMIT_BAL", 0), ("PAY_0", 9), ("PAY_AMT1", -1), ("MARRIAGE", 4)],
)
def test_rejects_out_of_range_values(client, applicant, field, value):
    applicant[field] = value
    assert client.post("/predict", json=applicant).status_code == 422


def test_rejects_wrong_type(client, applicant):
    applicant["AGE"] = "old"
    assert client.post("/predict", json=applicant).status_code == 422


def test_batch_limits(client, applicant):
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422
    too_many = {"applicants": [applicant] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_app_fails_fast_when_model_is_missing(tmp_path):
    app = create_app(tmp_path / "nope.joblib", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass
