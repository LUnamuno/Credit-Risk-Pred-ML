"""Data access, validation and splitting for the UCI Default of Credit Card Clients dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

UCI_DATASET_ID = 350
EXPECTED_ROWS = 30_000
TARGET = "default"
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "default_credit.csv"

# ucimlrepo exposes anonymous names X1..X23; these follow the dataset documentation.
FEATURE_RENAMES = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    **{f"X{6 + i}": name for i, name in enumerate(f"PAY_{m}" for m in (0, 2, 3, 4, 5, 6))},
    **{f"X{12 + i}": f"BILL_AMT{i + 1}" for i in range(6)},
    **{f"X{18 + i}": f"PAY_AMT{i + 1}" for i in range(6)},
}
FEATURES = list(FEATURE_RENAMES.values())
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]


class DatasetValidationError(ValueError):
    """Raised when the dataset does not look like the expected UCI table."""


def download_dataset(path: Path = RAW_FILE) -> Path:
    """Fetch the dataset from UCI via ``ucimlrepo`` and store it as a CSV with readable names."""
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=UCI_DATASET_ID)
    features = repo.data.features.rename(columns=FEATURE_RENAMES)
    target = repo.data.targets.iloc[:, 0].rename(TARGET)
    df = pd.concat([features, target], axis=1)
    validate_dataset(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def validate_dataset(df: pd.DataFrame, expected_rows: int | None = EXPECTED_ROWS) -> None:
    """Check shape, columns, missing values and target values.

    Pass ``expected_rows=None`` to skip the row-count check (e.g. for small test fixtures).
    """
    expected_cols = [*FEATURES, TARGET]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(f"Missing columns: {missing_cols}")
    if expected_rows is not None and len(df) != expected_rows:
        raise DatasetValidationError(f"Expected {expected_rows} rows, got {len(df)}")
    n_missing = int(df[expected_cols].isna().sum().sum())
    if n_missing:
        raise DatasetValidationError(f"Dataset contains {n_missing} missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DatasetValidationError("Target must be binary (0/1)")
    if df[TARGET].nunique() < 2:
        raise DatasetValidationError("Target has a single class")


def load_dataset(path: Path = RAW_FILE, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Load and validate the raw CSV, downloading it first if the default file is absent."""
    path = Path(path)
    if not path.exists():
        if path != RAW_FILE:
            raise FileNotFoundError(path)
        download_dataset(path)
    df = pd.read_csv(path)
    validate_dataset(df, expected_rows=expected_rows)
    return df


def split_data(
    df: pd.DataFrame,
    seed: int = SEED,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Stratified, reproducible train/validation/test split.

    The test split is carved out first and is only for the final evaluation: it must not be used
    to choose a model, a calibration method or a decision threshold.
    """
    X, y = df[FEATURES], df[TARGET]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_size / (1 - test_size),
        stratify=y_rest,
        random_state=seed,
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
