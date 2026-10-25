# Credit Risk Prediction: End-to-End ML

A small, reproducible machine-learning project that predicts whether a credit card client will
default next month. It covers the full loop: download data, validate it, train a scikit-learn
pipeline, calibrate its probabilities, turn them into a cost-based approve/reject decision,
evaluate on a held-out test set, and serve it through a FastAPI service in Docker, with tests and
CI from the first commit.

> **Disclaimer.** This is a portfolio project inspired by common credit-risk problems. It is **not**
> a reproduction of any bank's models, policies or data, and it is **not** a valid system for
> approving real credit. The cost assumptions behind the decision threshold are explicit,
> illustrative assumptions (the dataset has no loan amounts, margins or recoveries).

## Problem

Given 23 predictors for a card holder (credit limit, demographics, six months of repayment status,
bill amounts and payment amounts), estimate the **probability of default in the next month**
(binary target). The model outputs a calibrated probability; a separate, cost-based threshold
turns it into an approve/reject decision. The threshold depends on assumed costs, so it is a
business choice and not a property of the model.

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
- **Sensitive variables:** `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are included in the MVP. They
  raise fairness and regulatory concerns; a with/without sensitivity analysis is the first task
  of Sprint 3 (see the roadmap).
- **Data quirks:** `EDUCATION` (0, 5, 6) and `MARRIAGE` (0) contain undocumented codes; they are
  kept as their own categories.
- **Illustrative costs:** the 5:1 cost ratio (approving a defaulter vs. rejecting a good client) is
  an assumption. The threshold and the cost savings below only hold under that assumption.
- **MVP model:** untuned Logistic Regression only, no fairness analysis, no drift monitoring, and
  the service has no authentication or rate limiting.

## Project layout

```
.github/workflows/ci.yml    Ruff + pytest + package build + Docker build and smoke test
Dockerfile                  API image (no data or model baked in; see "Docker")
data/raw/                   downloaded CSV (gitignored)
examples/applicant.json     sample request body for the API
models/                     pipeline.joblib, decision.json, metrics.json (gitignored)
notebooks/01_eda.ipynb      minimal EDA (target, missing values, key variables)
scripts/download_data.py    fetch + validate the dataset
src/credit_risk/data.py     schema, download, validation, stratified split
src/credit_risk/train.py    pipeline, calibration choice, threshold, test evaluation, CLI
src/credit_risk/calibration.py  calibration methods and expected calibration error
src/credit_risk/decision.py     cost assumptions, threshold search, decision metrics
src/credit_risk/predict.py  load model/policy, predict_proba
src/credit_risk/api.py      FastAPI + Pydantic service
tests/                      offline tests (synthetic fixture), incl. API integration tests
```

## Reproduce the results

Requires Python 3.11+ and network access to UCI for the first run.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,api]"        # add ",notebook" to run the EDA notebook

python scripts/download_data.py    # optional: train downloads the data if it is missing
python -m credit_risk.train        # fit, calibrate, choose threshold, evaluate, save everything
```

Outputs in `models/`: `pipeline.joblib` (the calibrated model), `decision.json` (threshold and
cost assumptions) and `metrics.json` (all metrics below). Everything is seeded (`--seed`, default
42), so the numbers are reproducible. To try other cost assumptions:
`python -m credit_risk.train --cost-fn 10 --cost-fp 1`.

Quality checks (these also run in CI):

```bash
ruff check . && ruff format --check .
pytest                              # offline, no network needed
```

Score applicants from Python:

```python
from credit_risk.data import FEATURES, load_dataset
from credit_risk.predict import load_decision, predict_proba

df = load_dataset()
proba = predict_proba(df[FEATURES].head(3))  # array of 3 probabilities in [0, 1]
load_decision().decide(proba)  # array of "approve" / "reject"
predict_proba(df[FEATURES].iloc[0].to_dict())  # a single applicant as a dict
```

## API

After training, start the service (it loads `models/` once at startup and fails fast if they are
missing):

```bash
uvicorn credit_risk.api:app --port 8000      # interactive docs at http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | status, version, active threshold and cost assumptions |
| `POST /predict` | one applicant (23 fields) -> `probability_of_default`, `decision`, `threshold` |
| `POST /predict/batch` | `{"applicants": [...]}` with 1 to 1,000 applicants |

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d @examples/applicant.json
# {"probability_of_default":0.68,"decision":"reject","threshold":0.18}
```

Requests are validated with Pydantic: missing or unknown fields, wrong types and out-of-range
values (e.g. `SEX` not in 1-2, `AGE` under 18, `PAY_*` outside -2..8) return HTTP 422. Override the
model location with `CREDIT_RISK_MODEL_PATH` and `CREDIT_RISK_DECISION_PATH`.

## Docker

The image contains no data and no model. Train inside the container (same library versions as the
API, so the saved model always loads) into mounted folders, then serve from them:

```bash
docker build -t credit-risk .
mkdir -p models data/raw
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/models:/app/models" \
    -v "$PWD/data:/app/data" credit-risk python -m credit_risk.train
docker run -p 8000:8000 --user "$(id -u):$(id -g)" -v "$PWD/models:/app/models:ro" credit-risk
```

On Windows PowerShell use `${PWD}` instead of `$PWD` and omit `--user`. CI builds the image,
trains inside it and smoke-tests `/health` and `/predict`.

## Method

- **Split:** stratified 60/20/20 train/validation/test with a fixed seed. The three splits have
  separate roles, and the test split never influences any choice:
  1. **train:** fit the model; calibration is learned here with internal 5-fold cross-validation.
  2. **validation:** choose the calibration method (lowest Brier score among none, sigmoid and
     isotonic) and the decision threshold (lowest average cost).
  3. **test:** one final evaluation of the already-fixed model and threshold. Do not iterate on
     it; any further tuning should use the validation split.
- **Pipeline:** `ColumnTransformer` (standard-scale numeric features, one-hot encode `SEX`,
  `EDUCATION`, `MARRIAGE` with unseen categories ignored) followed by `LogisticRegression`.
  All preprocessing lives inside the `Pipeline` and is fitted on the training split only, so there
  is no leakage.
- **Calibration:** `none`, `sigmoid` (Platt) and `isotonic` are compared by Brier score and
  expected calibration error (ECE, 10 equal-width bins). The best one on validation is saved.
- **Decision:** reject when `P(default) >= threshold`. With costs `c_FN` (approve a defaulter) and
  `c_FP` (reject a client who would have repaid), a perfectly calibrated model would use
  `c_FP / (c_FP + c_FN)`; here 1/6 = 0.167. The saved threshold is searched on a 0.01 grid over
  validation probabilities.
- **Swappable model:** `MODEL_FACTORIES` in `train.py` maps names to estimator factories, so
  Random Forest or gradient boosting can later be compared under the same split, preprocessing,
  calibration and metrics (`python -m credit_risk.train --model <name>`).
- **Metrics:** ROC-AUC, PR-AUC (average precision), Brier score and ECE. Accuracy is not reported
  because of the class imbalance.

## Results (MVP: Logistic Regression)

Seed 42; 6,000 validation rows and 6,000 test rows (22.1% defaults each).

**Calibration (validation).** Raw Logistic Regression is already decent, but isotonic
calibration cuts the calibration error by about 7x and the Brier score by about 3%.

| Calibration | Brier | ECE | ROC-AUC |
| --- | --- | --- | --- |
| none | 0.1452 | 0.046 | 0.726 |
| sigmoid | 0.1454 | 0.048 | 0.726 |
| **isotonic (selected)** | **0.1409** | **0.006** | 0.730 |

**Held-out test set** (the numbers to quote; validation values are slightly optimistic because
the calibration method was picked on them):

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.714 |
| PR-AUC | 0.494 |
| Brier score | 0.142 |
| ECE | 0.007 |

For reference, a model that always predicts the base rate has PR-AUC about 0.22 and Brier about
0.172.

**Decision on the test set** (threshold 0.18, costs 5:1, average cost per applicant in relative
units, lower is better):

| Policy | Cost per applicant |
| --- | --- |
| Approve everyone | 1.106 |
| Reject everyone | 0.779 |
| Threshold 0.5 | 0.760 |
| **Model, threshold 0.18** | **0.642** |

At the chosen threshold the service approves 68.4% of applicants, rejects 58.8% of the real
defaulters, and 41.1% of its rejections are real defaulters (the rest are good clients lost). The
cost is about 15% lower than the best naive policy, **under the assumed 5:1 cost ratio**. With
other ratios the threshold and the savings change. All numbers are written to
`models/metrics.json` by every training run.

## Roadmap

1. **Sprint 1 (done):** data, baseline pipeline, metrics, tests, CI.
2. **Sprint 2 (done, MVP with Logistic Regression):** calibration, thresholds and cost-sensitive
   decisions (with explicit cost assumptions), first held-out test-set evaluation, FastAPI +
   Pydantic service, Docker, integration tests. The MVP is complete after this sprint.
3. **Sprint 3 (Expansion):** sensitivity analysis without sensitive variables, then Random Forest
   and HistGradientBoosting/XGBoost compared with Logistic Regression under the same split,
   preprocessing and metrics (including the cost metric from Sprint 2).
4. **Sprint 4 (Polish):** drift report, model card, extended CI/CD, demo.

> **IMPORTANT NOTE.** The sensitivity analysis (with vs. without `SEX`, `EDUCATION`, `MARRIAGE`
> and `AGE`) is a priority and must be done **first in Sprint 3, before comparing other models**.
> It is cheap with Logistic Regression, and if the sensitive variables carry a lot of weight it
> may change which features the MVP keeps.
