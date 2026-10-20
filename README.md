# Credit Risk Prediction: End-to-End ML

A small, reproducible machine-learning project that predicts whether a credit card client will
default next month. It covers the full loop: download data, validate it, train a scikit-learn
pipeline, evaluate, persist the model, score new applicants, with tests and CI from the first commit.

> **Disclaimer.** This is a portfolio project inspired by common credit-risk problems. It is **not**
> a reproduction of any bank's models, policies or data, and it is **not** a valid system for
> approving real credit. Any cost assumptions added in later sprints are explicit assumptions.

## Problem

Given 23 predictors for a card holder (credit limit, demographics, six months of repayment status,
bill amounts and payment amounts), estimate the **probability of default in the next month**
(binary target). The output is a probability (calibration is not yet verified), not a hard decision; choosing a
decision threshold is a business/cost question handled in Sprint 2.

**Data:** [UCI "Default of Credit Card Clients"](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(id=350): 30,000 rows, 23 predictors, 22.1% positive class. Fetched with
[`ucimlrepo`](https://github.com/uci-ml-repo/ucimlrepo) into `data/raw/` (gitignored).

## Limitations

- **Old, narrow portfolio:** a single Taiwanese card issuer in 2005. Results do not transfer to
  other markets or today's customers.
- **No per-row dates:** the table has no application/observation date, so a realistic
  out-of-time validation is impossible. The split is random (stratified), which is optimistic
  compared with a temporal split.
- **Short horizon:** the label is default in the *next month* only.
- **Sensitive variables:** `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are included in Sprint 1. They
  raise fairness and regulatory concerns; a with/without sensitivity analysis is planned.
- **Data quirks:** `EDUCATION` (0, 5, 6) and `MARRIAGE` (0) contain undocumented codes; they are
  kept as their own categories.
- **Sprint 1 is a baseline:** untuned Logistic Regression, no calibration analysis, no threshold
  or cost analysis, no fairness analysis.

## Project layout

```
.github/workflows/ci.yml   Ruff + pytest + package build
data/raw/                  downloaded CSV (gitignored)
models/                    saved pipeline and metrics.json (gitignored)
notebooks/01_eda.ipynb     minimal EDA (target, missing values, key variables)
scripts/download_data.py   fetch + validate the dataset
src/credit_risk/data.py    schema, download, validation, stratified split
src/credit_risk/train.py   pipeline, model factory, metrics, CLI (python -m credit_risk.train)
src/credit_risk/predict.py load pipeline, predict_proba
tests/                     offline tests using a small synthetic fixture
```

## Reproduce the results

Requires Python 3.11+ and network access to UCI for the first run.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",notebook" to run the EDA notebook

python scripts/download_data.py    # optional: train downloads the data if it is missing
python -m credit_risk.train        # fit on train, evaluate on validation, save model + metrics
```

Outputs: `models/pipeline.joblib` and `models/metrics.json`. Everything is seeded (`--seed`, default
42), so the metrics below are reproducible.

Quality checks (these also run in CI):

```bash
ruff check . && ruff format --check .
pytest                              # offline, no network needed
```

Score applicants with the saved pipeline:

```python
from credit_risk.predict import predict_proba

predict_proba({"LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
               "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
               "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000,
               "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
               "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
               "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000})
# -> array([p]) with p in [0, 1]
```

## Method

- **Split:** stratified 60/20/20 train/validation/test with a fixed seed. The test set is held out
  and **not used** in Sprint 1.
- **Pipeline:** `ColumnTransformer` (standard-scale numeric features, one-hot encode `SEX`,
  `EDUCATION`, `MARRIAGE` with unseen categories ignored) followed by `LogisticRegression`.
  All preprocessing lives inside the `Pipeline` and is fitted on the training split only, so there
  is no leakage.
- **Swappable model:** `MODEL_FACTORIES` in `train.py` maps names to estimator factories, so
  Random Forest or gradient boosting can later be compared under the same split, preprocessing and
  metrics (`python -m credit_risk.train --model <name>`).
- **Metrics (validation):** ROC-AUC, PR-AUC (average precision) and Brier score. Accuracy is not
  reported because of the class imbalance.

## Results (Sprint 1 baseline)

Logistic Regression, seed 42, 6,000 validation rows (22.1% defaults):

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.726 |
| PR-AUC | 0.500 |
| Brier score | 0.145 |

For reference, a model that always predicts the base rate has PR-AUC about 0.22 and Brier about
0.172. Numbers are written to `models/metrics.json` by every training run.

## Roadmap

1. **Sprint 1 (this):** data, baseline pipeline, metrics, tests, CI.
2. **Sprint 2:** calibration, thresholds and cost-sensitive decisions, Random Forest and
   HistGradientBoosting/XGBoost, sensitivity analysis without sensitive variables.
3. **Sprint 3:** FastAPI + Pydantic service, Docker, integration tests.
4. **Sprint 4:** drift report, model card, extended CI/CD, demo.
