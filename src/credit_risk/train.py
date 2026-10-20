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

from credit_risk.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FILE,
    ROOT,
    SEED,
    load_dataset,
    split_data,
)

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


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
) -> dict[str, Any]:
    """Fit the pipeline on the train split, evaluate on validation, persist model and metrics."""
    df = load_dataset(data_path, expected_rows=expected_rows)
    splits = split_data(df, seed=seed)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    pipeline = build_pipeline(model_name, seed)
    pipeline.fit(X_train, y_train)
    val_metrics = compute_metrics(y_val, pipeline.predict_proba(X_val)[:, 1])

    report = {
        "model": model_name,
        "seed": seed,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(splits["test"][0]),
        "validation_default_rate": float(y_val.mean()),
        "validation": val_metrics,
    }
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_FILE, help="Raw CSV (auto-downloaded).")
    parser.add_argument("--model", default="logistic_regression", choices=sorted(MODEL_FACTORIES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    report = train(args.data, args.model, args.seed, args.model_path, args.metrics_path)
    print(json.dumps(report, indent=2))
    print(f"Saved model to {args.model_path} and metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
