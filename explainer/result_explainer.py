"""Human-readable explanation for a software effort estimation result.

Covers:
- Why this estimate was produced (dominant dimension)
- Which PM input fields most drove up the cost
- Reliability classification based on confidence_pct
"""
from __future__ import annotations

import dataclasses

from core.schema import PMInput
from core.estimator_bridge import BridgeResult

# Field labels shown to the PM in explanation text
_FIELD_LABELS: dict[str, str] = {
    "num_features": "number of features",
    "feature_complexity": "feature complexity",
    "num_apis": "number of API endpoints",
    "platform": "target platform",
    "third_party_integrations": "third-party integrations",
    "security_level": "security requirements",
    "num_developers": "team size",
    "seniority": "team seniority",
    "stack_experience": "stack experience",
    "deadline_months": "project deadline",
    "budget_constraint": "budget constraint",
    "requirement_change_risk": "requirement change risk",
    "uncertainty_level": "project uncertainty",
    "technical_debt": "technical debt",
    "external_dependencies": "external dependencies",
}

# Fields ordered by their typical effort-driving impact; each entry includes
# how a "high-risk" value looks for that field
_HIGH_RISK_VALUES: dict[str, set] = {
    "feature_complexity": {"high"},
    "security_level": {"high"},
    "requirement_change_risk": {"high"},
    "uncertainty_level": {"high"},
    "technical_debt": {"high"},
    "budget_constraint": {"hard"},
    "seniority": {"junior"},
    "stack_experience": {"low"},
}

_HIGH_RISK_NUMERIC: dict[str, tuple] = {
    # (threshold, label_for_exceeding)
    "num_features": (15, "large feature count"),
    "num_apis": (8, "many API endpoints"),
    "third_party_integrations": (3, "several third-party integrations"),
    "external_dependencies": (3, "many external dependencies"),
}


def _reliability_label(confidence_pct: float) -> str:
    if confidence_pct >= 80:
        return "High"
    if confidence_pct >= 50:
        return "Moderate"
    return "Low"


def _top_cost_drivers(pm_input: PMInput, top_n: int = 3) -> list[str]:
    """Return up to top_n field labels that are in a high-risk state."""
    drivers: list[str] = []
    fields = dataclasses.asdict(pm_input)

    for field_name, high_values in _HIGH_RISK_VALUES.items():
        value = fields.get(field_name)
        if value in high_values:
            drivers.append(_FIELD_LABELS[field_name])

    for field_name, (threshold, label) in _HIGH_RISK_NUMERIC.items():
        value = fields.get(field_name, 0)
        if isinstance(value, (int, float)) and value > threshold:
            drivers.append(label)

    return drivers[:top_n]


def explain(
    pm_input: PMInput,
    result: BridgeResult,
    dominance: dict,
    confidence: dict,
) -> str:
    """Return a human-readable explanation of the estimation result.

    Parameters
    ----------
    pm_input   : the PM's project parameters
    result     : BridgeResult from estimator_bridge.estimate()
    dominance  : dict from dominance.compute()
    confidence : dict from confidence.compute()
    """
    effort = result.effort_pm
    model = result.model_info
    conf_pct = confidence["confidence_pct"]
    lower = confidence["lower"]
    upper = confidence["upper"]

    # ── Dominant dimension ────────────────────────────────────────────────────
    dominant_dim = max(dominance, key=dominance.get)
    dominant_share = dominance[dominant_dim]

    # ── Reliability ───────────────────────────────────────────────────────────
    reliability = _reliability_label(conf_pct)
    reliability_caveat = {
        "High": "The estimate is well-supported by the training data.",
        "Moderate": "The estimate is reasonable but carries some uncertainty.",
        "Low": "Warning: high uncertainty — treat this estimate as a rough guide only.",
    }[reliability]

    # ── Cost drivers ──────────────────────────────────────────────────────────
    drivers = _top_cost_drivers(pm_input)
    if drivers:
        driver_text = "Key cost drivers: " + ", ".join(drivers) + "."
    else:
        driver_text = "No high-risk factors detected; the estimate reflects a low-complexity project."

    # ── Compose narrative ─────────────────────────────────────────────────────
    lines = [
        f"Estimated effort: {effort:.1f} person-months",
        f"  Range: {lower:.1f} - {upper:.1f} person-months (+/-{confidence['range_pct']:.0f}%)",
        "",
        f"Reliability: {reliability} ({conf_pct:.0f}% confidence).",
        reliability_caveat,
        "",
        f"Dominant dimension: {dominant_dim} ({dominant_share:.0f}% of effort).",
        "Effort breakdown - "
        + ", ".join(f"{dim}: {share:.0f}%" for dim, share in sorted(dominance.items(), key=lambda x: -x[1])),
        "",
        driver_text,
        "",
        f"Model: {model.model_name} trained on {model.stem} dataset "
        f"({model.trained_on_rows} projects). CV MAE: {model.cv_mae_pm:.1f} person-months.",
    ]

    return "\n".join(lines)
