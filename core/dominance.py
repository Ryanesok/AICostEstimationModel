"""Effort dominance breakdown by PM parameter dimension.

Each dimension (Technical, Team, Business, Risk) receives a raw score
computed from its constituent PM fields. Scores are normalised to 100%.

Weighting rationale
-------------------
Technical (features, complexity, APIs, integrations, security):
  score = num_features * complexity_w + num_apis * 2 + integrations * 1.5 + security_w
  Complexity and security are major cost amplifiers in FP-based literature.

Team (developers, seniority, stack experience):
  score = num_developers * (1 / seniority_w) * (1 / stack_w)
  More developers and lower seniority / stack experience drive up effort.

Business (deadline, budget, requirement change risk):
  score = (1 / deadline_months) * 10 + budget_w + change_risk_w
  Short deadlines and tight budgets force overtime; requirement churn wastes effort.

Risk (uncertainty, technical debt, external dependencies):
  score = uncertainty_w + debt_w + external_deps * 1.5
  Uncertainty and technical debt are well-documented effort multipliers (cf. COCOMO II).
"""
from __future__ import annotations

from core.schema import PMInput
from core.estimator_bridge import BridgeResult

# Ordinal weights referenced in the formulae above
_COMPLEXITY_W = {"low": 1.0, "medium": 2.0, "high": 4.0}
_SECURITY_W = {"basic": 0.5, "standard": 1.5, "high": 3.5}
_SENIORITY_W = {"junior": 0.5, "mid": 1.0, "senior": 2.0, "mixed": 0.8}
_STACK_W = {"low": 0.5, "medium": 1.0, "high": 2.0}
_BUDGET_W = {"none": 0.5, "soft": 1.5, "hard": 3.5}
_CHANGE_W = {"low": 0.5, "medium": 2.0, "high": 4.0}
_UNCERTAINTY_W = {"low": 1.0, "medium": 2.5, "high": 5.0}
_DEBT_W = {"none": 0.5, "low": 2.0, "high": 5.0}


def _score_technical(pm: PMInput) -> float:
    return (
        pm.num_features * _COMPLEXITY_W[pm.feature_complexity]
        + pm.num_apis * 2.0
        + pm.third_party_integrations * 1.5
        + _SECURITY_W[pm.security_level]
    )


def _score_team(pm: PMInput) -> float:
    # Low seniority/stack experience means more effort per developer
    inv_seniority = 1.0 / _SENIORITY_W[pm.seniority]
    inv_stack = 1.0 / _STACK_W[pm.stack_experience]
    return pm.num_developers * inv_seniority * inv_stack


def _score_business(pm: PMInput) -> float:
    # Short deadlines add schedule pressure proportional to 1/months
    return (1.0 / max(pm.deadline_months, 0.1)) * 10.0 + _BUDGET_W[pm.budget_constraint] + _CHANGE_W[pm.requirement_change_risk]


def _score_risk(pm: PMInput) -> float:
    return _UNCERTAINTY_W[pm.uncertainty_level] + _DEBT_W[pm.technical_debt] + pm.external_dependencies * 1.5


def compute(pm_input: PMInput, result: BridgeResult) -> dict:
    """Return effort share (%) per dimension, summing to 100.

    Keys: 'Technical', 'Team', 'Business', 'Risk'
    """
    scores = {
        "Technical": _score_technical(pm_input),
        "Team": _score_team(pm_input),
        "Business": _score_business(pm_input),
        "Risk": _score_risk(pm_input),
    }

    # Weight by effort_pm to anchor to the actual prediction magnitude
    # (All dimensions share the same effort_pm, so this cancels out —
    #  the split is purely from the relative scores.)
    total = sum(scores.values())
    if total == 0:
        equal = 25.0
        return {"Technical": equal, "Team": equal, "Business": equal, "Risk": equal}

    return {dim: round(score / total * 100.0, 2) for dim, score in scores.items()}
