"""FastAPI service: ``uvicorn credit_risk.api:app``.

Loads the model and decision policy produced by ``python -m credit_risk.train`` once at startup.
Paths can be overridden with ``CREDIT_RISK_MODEL_PATH`` and ``CREDIT_RISK_DECISION_PATH``.

Portfolio demo only: it is not a valid system for approving real credit.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from credit_risk import __version__
from credit_risk.data import FEATURES
from credit_risk.predict import load_decision, load_pipeline, predict_proba
from credit_risk.train import DECISION_PATH, MODEL_PATH

MAX_BATCH_SIZE = 1000
# Repayment status codes in the UCI table: -2..8 (-2 no consumption, -1 paid in full, 0 revolving
# credit, 1..8 months of payment delay).
PayStatus = Annotated[int, Field(ge=-2, le=8)]
Amount = Annotated[float, Field(description="New Taiwan dollars (NT$)")]
NonNegativeAmount = Annotated[float, Field(ge=0, description="New Taiwan dollars (NT$)")]


class Applicant(BaseModel):
    """The 23 predictors of the UCI table. Field names match ``credit_risk.data.FEATURES``."""

    model_config = ConfigDict(extra="forbid")

    LIMIT_BAL: Annotated[float, Field(gt=0, description="Credit limit in NT$")]
    SEX: Annotated[int, Field(ge=1, le=2, description="1 = male, 2 = female")]
    EDUCATION: Annotated[
        int, Field(ge=0, le=6, description="1 grad school, 2 university, 3 high school, 4 other")
    ]
    MARRIAGE: Annotated[int, Field(ge=0, le=3, description="1 married, 2 single, 3 other")]
    AGE: Annotated[int, Field(ge=18, le=120)]
    PAY_0: PayStatus
    PAY_2: PayStatus
    PAY_3: PayStatus
    PAY_4: PayStatus
    PAY_5: PayStatus
    PAY_6: PayStatus
    BILL_AMT1: Amount
    BILL_AMT2: Amount
    BILL_AMT3: Amount
    BILL_AMT4: Amount
    BILL_AMT5: Amount
    BILL_AMT6: Amount
    PAY_AMT1: NonNegativeAmount
    PAY_AMT2: NonNegativeAmount
    PAY_AMT3: NonNegativeAmount
    PAY_AMT4: NonNegativeAmount
    PAY_AMT5: NonNegativeAmount
    PAY_AMT6: NonNegativeAmount


class Prediction(BaseModel):
    probability_of_default: float = Field(ge=0, le=1)
    decision: Literal["approve", "reject"]
    threshold: float = Field(description="Reject when probability_of_default >= threshold")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicants: Annotated[list[Applicant], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    threshold: float
    cost_false_negative: float
    cost_false_positive: float


def create_app(
    model_path: Path | None = None,
    decision_path: Path | None = None,
) -> FastAPI:
    """Build the app. Model and policy are loaded when the server starts, not at import time."""
    model_path = Path(model_path or os.environ.get("CREDIT_RISK_MODEL_PATH", MODEL_PATH))
    decision_path = Path(
        decision_path or os.environ.get("CREDIT_RISK_DECISION_PATH", DECISION_PATH)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = load_pipeline(model_path)
        app.state.policy = load_decision(decision_path)
        yield

    app = FastAPI(
        title="Credit Risk Prediction API",
        version=__version__,
        description=(
            "Probability of default next month for a credit card client, plus an approve/reject "
            "decision from a cost-based threshold. **Portfolio demo: not a valid system for "
            "approving real credit.** Costs are illustrative assumptions."
        ),
        lifespan=lifespan,
    )

    def score(request: Request, applicants: list[Applicant]) -> list[Prediction]:
        model, policy = request.app.state.model, request.app.state.policy
        frame = pd.DataFrame([a.model_dump() for a in applicants])[FEATURES]
        try:
            proba = predict_proba(frame, model)
        except Exception as exc:  # pragma: no cover - defensive, inputs are already validated
            raise HTTPException(status_code=500, detail="Scoring failed") from exc
        decisions = policy.decide(proba)
        return [
            Prediction(probability_of_default=float(p), decision=str(d), threshold=policy.threshold)
            for p, d in zip(proba, decisions, strict=True)
        ]

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        policy = request.app.state.policy
        return Health(
            status="ok",
            version=__version__,
            threshold=policy.threshold,
            cost_false_negative=policy.costs.cost_false_negative,
            cost_false_positive=policy.costs.cost_false_positive,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(applicant: Applicant, request: Request) -> Prediction:
        return score(request, [applicant])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
        return BatchResponse(predictions=score(request, body.applicants))

    return app


app = create_app()
