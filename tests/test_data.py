from __future__ import annotations

import pytest

from credit_risk.data import FEATURES, TARGET, DatasetValidationError, split_data, validate_dataset


def test_fixture_schema_is_valid(synthetic_df):
    validate_dataset(synthetic_df, expected_rows=None)
    assert len(FEATURES) == 23


def test_validation_rejects_wrong_row_count(synthetic_df):
    with pytest.raises(DatasetValidationError, match="rows"):
        validate_dataset(synthetic_df)


def test_validation_rejects_missing_values(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "AGE"] = None
    with pytest.raises(DatasetValidationError, match="missing"):
        validate_dataset(bad, expected_rows=None)


def test_validation_rejects_non_binary_target(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, TARGET] = 2
    with pytest.raises(DatasetValidationError, match="binary"):
        validate_dataset(bad, expected_rows=None)


def test_validation_rejects_missing_columns(synthetic_df):
    with pytest.raises(DatasetValidationError, match="columns"):
        validate_dataset(synthetic_df.drop(columns=["AGE"]), expected_rows=None)


def test_split_is_stratified_disjoint_and_reproducible(synthetic_df):
    a = split_data(synthetic_df, seed=1)
    b = split_data(synthetic_df, seed=1)
    assert set(a) == {"train", "val", "test"}
    for name in a:
        assert a[name][0].index.equals(b[name][0].index)
    indexes = [set(a[name][0].index) for name in a]
    assert sum(len(i) for i in indexes) == len(synthetic_df)
    assert set.intersection(*indexes) == set()
    overall = synthetic_df[TARGET].mean()
    for name in a:
        assert a[name][1].mean() == pytest.approx(overall, abs=0.02)
    assert len(a["train"][0]) == pytest.approx(0.6 * len(synthetic_df), abs=2)
