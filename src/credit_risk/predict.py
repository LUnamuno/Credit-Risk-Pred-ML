"""Load the trained pipeline and score applicants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.data import FEATURES
from credit_risk.train import MODEL_PATH


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    """Load a pipeline saved by ``credit_risk.train``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m credit_risk.train`.")
    return joblib.load(path)


def predict_proba(
    records: pd.DataFrame | Mapping[str, float] | Sequence[Mapping[str, float]],
    pipeline: Pipeline | None = None,
) -> np.ndarray:
    """Return the probability of default (in [0, 1]) for each input row.

    ``records`` may be a DataFrame, a single mapping or a list of mappings containing the 23
    predictor columns defined in ``credit_risk.data.FEATURES``.
    """
    if isinstance(records, pd.DataFrame):
        frame = records
    elif isinstance(records, Mapping):
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(list(records))

    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    pipeline = pipeline if pipeline is not None else load_pipeline()
    return pipeline.predict_proba(frame[FEATURES])[:, 1]
