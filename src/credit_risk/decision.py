"""Cost-sensitive decision rule on top of calibrated default probabilities.

Convention: an applicant is **rejected** when ``P(default) >= threshold`` and **approved**
otherwise. Costs are expressed in arbitrary, relative units.

**The default costs are assumptions, not data.** The UCI table has no loan amounts, margins or
recoveries, so the 5:1 ratio below is an illustrative choice: approving a client who defaults is
assumed to cost five times as much as rejecting a client who would have repaid. Change the
ratio (``--cost-fn`` / ``--cost-fp`` in the training CLI) to see how the decision moves.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

APPROVE = "approve"
REJECT = "reject"
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


@dataclass(frozen=True)
class CostAssumptions:
    """Relative costs of the two kinds of wrong decision."""

    cost_false_negative: float = 5.0  # approved an applicant who then defaults
    cost_false_positive: float = 1.0  # rejected an applicant who would have repaid

    def __post_init__(self) -> None:
        if self.cost_false_negative <= 0 or self.cost_false_positive <= 0:
            raise ValueError("Costs must be positive")

    @property
    def theoretical_threshold(self) -> float:
        """Cost-minimising threshold if the probabilities were perfectly calibrated."""
        return self.cost_false_positive / (self.cost_false_positive + self.cost_false_negative)


DEFAULT_COSTS = CostAssumptions()


@dataclass(frozen=True)
class DecisionPolicy:
    """Threshold plus the cost assumptions it was chosen under (saved next to the model)."""

    threshold: float
    costs: CostAssumptions

    def decide(self, y_proba: Any) -> np.ndarray:
        """Map probabilities to ``"approve"`` / ``"reject"`` labels."""
        reject = np.asarray(y_proba, dtype=float) >= self.threshold
        return np.where(reject, REJECT, APPROVE)

    def save(self, path: Path) -> None:
        payload = {"threshold": self.threshold, **asdict(self.costs)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> DecisionPolicy:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No decision policy at {path}. Run `python -m credit_risk.train`."
            )
        data = json.loads(path.read_text())
        costs = CostAssumptions(data["cost_false_negative"], data["cost_false_positive"])
        return cls(threshold=float(data["threshold"]), costs=costs)


def confusion_counts(y_true: Any, y_proba: Any, threshold: float) -> dict[str, int]:
    """Counts with "reject" as the positive class (predicted default)."""
    y = np.asarray(y_true).astype(bool)
    reject = np.asarray(y_proba, dtype=float) >= threshold
    return {
        "tp": int((reject & y).sum()),  # rejected, would have defaulted
        "fp": int((reject & ~y).sum()),  # rejected, would have repaid
        "fn": int((~reject & y).sum()),  # approved, defaulted
        "tn": int((~reject & ~y).sum()),  # approved, repaid
    }


def expected_cost(
    y_true: Any, y_proba: Any, threshold: float, costs: CostAssumptions = DEFAULT_COSTS
) -> float:
    """Average cost per applicant at ``threshold``."""
    c = confusion_counts(y_true, y_proba, threshold)
    total = costs.cost_false_negative * c["fn"] + costs.cost_false_positive * c["fp"]
    return float(total / len(np.asarray(y_true)))


def find_best_threshold(y_true: Any, y_proba: Any, costs: CostAssumptions = DEFAULT_COSTS) -> float:
    """Threshold on a 0.01 grid with the lowest average cost (earliest one on ties)."""
    grid_costs = [expected_cost(y_true, y_proba, t, costs) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmin(grid_costs))])


def evaluate_decision(
    y_true: Any, y_proba: Any, threshold: float, costs: CostAssumptions = DEFAULT_COSTS
) -> dict[str, float | int]:
    """Cost, confusion counts and rates for a decision rule on one split."""
    c = confusion_counts(y_true, y_proba, threshold)
    n = sum(c.values())
    n_default = c["tp"] + c["fn"]
    n_reject = c["tp"] + c["fp"]
    return {
        "threshold": float(threshold),
        "cost_per_applicant": expected_cost(y_true, y_proba, threshold, costs),
        "approval_rate": (c["tn"] + c["fn"]) / n,
        # Share of the real defaulters that were rejected / share of rejections that were right.
        "default_capture_rate": c["tp"] / n_default if n_default else 0.0,
        "rejection_precision": c["tp"] / n_reject if n_reject else 0.0,
        **c,
    }


def baseline_costs(
    y_true: Any, y_proba: Any, costs: CostAssumptions = DEFAULT_COSTS
) -> dict[str, float]:
    """Average cost of naive policies, to judge whether the model adds value."""
    return {
        "approve_everyone": expected_cost(y_true, y_proba, 1.01, costs),
        "reject_everyone": expected_cost(y_true, y_proba, 0.0, costs),
        "threshold_0.5": expected_cost(y_true, y_proba, 0.5, costs),
    }
