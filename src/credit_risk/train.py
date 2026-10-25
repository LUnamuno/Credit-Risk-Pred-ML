"""Training entry point: ``python -m credit_risk.train``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.calibration import (
    CALIBRATION_METHODS,
    build_calibrated,
    expected_calibration_error,
)
from credit_risk.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FILE,
    ROOT,
    SEED,
    load_dataset,
    split_data,
)
from credit_risk.decision import (
    DEFAULT_COSTS,
    CostAssumptions,
    DecisionPolicy,
    baseline_costs,
    evaluate_decision,
    find_best_threshold,
)

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
DECISION_FILENAME = "decision.json"
DECISION_PATH = MODELS_DIR / DECISION_FILENAME


def _logistic_regression(seed: int) -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=seed)


# Register new estimators here (e.g. RandomForest, HistGradientBoosting) to compare them
# under the exact same split, preprocessing and metrics.
MODEL_FACTORIES: dict[str, Callable[[int], BaseEstimator]] = {
    "logistic_regression": _logistic_regression,
}


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones (unseen levels ignored)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model_name: str = "logistic_regression", seed: int = SEED) -> Pipeline:
    """Preprocessing + estimator in one Pipeline so all fitting happens on training data only."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_FACTORIES)}")
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", MODEL_FACTORIES[model_name](seed)),
        ]
    )


def compute_metrics(y_true: pd.Series, y_proba: Any) -> dict[str, float]:
    """ROC-AUC, PR-AUC (average precision) and Brier score for predicted default probabilities."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def train(
    data_path: Path = RAW_FILE,
    model_name: str = "logistic_regression",
    seed: int = SEED,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    expected_rows: int | None = 30_000,
    costs: CostAssumptions = DEFAULT_COSTS,
    decision_path: Path | None = None,
) -> dict[str, Any]:
    """Fit, calibrate, choose a threshold, evaluate on the test split and persist everything.

    Order of use of the splits (the test split never influences any choice):

    1. **train:** fit the model; calibration is learned here via internal cross-validation.
    2. **validation:** choose the calibration method (lowest Brier score) and the decision
       threshold (lowest expected cost under ``costs``).
    3. **test:** one final evaluation of the already-fixed model and threshold.

    ``decision_path`` defaults to ``decision.json`` next to ``metrics_path``.
    """
    df = load_dataset(data_path, expected_rows=expected_rows)
    splits = split_data(df, seed=seed)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # 1-2. Fit every calibration option on train and compare them on validation.
    fitted: dict[str, BaseEstimator] = {}
    val_proba: dict[str, Any] = {}
    calibration_val: dict[str, dict[str, float]] = {}
    for method in CALIBRATION_METHODS:
        candidate = build_calibrated(build_pipeline(model_name, seed), method, seed)
        candidate.fit(X_train, y_train)
        proba = candidate.predict_proba(X_val)[:, 1]
        fitted[method], val_proba[method] = candidate, proba
        calibration_val[method] = {
            **compute_metrics(y_val, proba),
            "ece": expected_calibration_error(y_val, proba),
        }
    selected = min(CALIBRATION_METHODS, key=lambda m: calibration_val[m]["brier"])
    model = fitted[selected]

    # 2. Decision threshold from validation probabilities of the selected model.
    threshold = find_best_threshold(y_val, val_proba[selected], costs)
    policy = DecisionPolicy(threshold=threshold, costs=costs)

    # 3. Final, one-off evaluation on the held-out test split.
    test_proba = model.predict_proba(X_test)[:, 1]

    report: dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "validation_default_rate": float(y_val.mean()),
        "test_default_rate": float(y_test.mean()),
        "validation": compute_metrics(y_val, val_proba[selected]),
        "calibration": {"selected": selected, "validation": calibration_val},
        "test": {
            **compute_metrics(y_test, test_proba),
            "ece": expected_calibration_error(y_test, test_proba),
        },
        "decision": {
            "assumptions": {
                "cost_false_negative": costs.cost_false_negative,
                "cost_false_positive": costs.cost_false_positive,
                "theoretical_threshold": costs.theoretical_threshold,
            },
            "threshold": threshold,
            "validation": evaluate_decision(y_val, val_proba[selected], threshold, costs),
            "test": evaluate_decision(y_test, test_proba, threshold, costs),
            "test_baselines_cost_per_applicant": baseline_costs(y_test, test_proba, costs),
        },
    }

    model_path, metrics_path = Path(model_path), Path(metrics_path)
    decision_path = (
        Path(decision_path) if decision_path else metrics_path.parent / DECISION_FILENAME
    )
    for path in (model_path, metrics_path, decision_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    policy.save(decision_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_FILE, help="Raw CSV (auto-downloaded).")
    parser.add_argument("--model", default="logistic_regression", choices=sorted(MODEL_FACTORIES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument(
        "--decision-path", type=Path, default=None, help="Default: decision.json next to metrics."
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=DEFAULT_COSTS.cost_false_negative,
        help="Assumed relative cost of approving an applicant who defaults.",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=DEFAULT_COSTS.cost_false_positive,
        help="Assumed relative cost of rejecting an applicant who would have repaid.",
    )
    args = parser.parse_args()

    report = train(
        args.data,
        args.model,
        args.seed,
        args.model_path,
        args.metrics_path,
        costs=CostAssumptions(args.cost_fn, args.cost_fp),
        decision_path=args.decision_path,
    )
    print(json.dumps(report, indent=2))
    print(f"Saved model to {args.model_path} and metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
