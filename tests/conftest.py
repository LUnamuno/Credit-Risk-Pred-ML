"""Shared fixtures: a small synthetic dataset with the same schema as the UCI table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import FEATURES, TARGET


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(0, 7, n),
            "MARRIAGE": rng.integers(0, 4, n),
            "AGE": rng.integers(21, 70, n),
        }
    )
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        df[col] = rng.integers(-2, 4, n)
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.integers(-5_000, 200_000, n)
        df[f"PAY_AMT{i}"] = rng.integers(0, 50_000, n)
    logit = 0.8 * df["PAY_0"] - 1.0 + rng.normal(0, 1, n)
    df[TARGET] = (logit > 0).astype(int)
    return df[[*FEATURES, TARGET]]
