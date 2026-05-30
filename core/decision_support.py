"""Decision support: overrun detection, resource optimization, risk advisory."""
from __future__ import annotations

import copy

from core.schema import PMInput
from core.estimator_bridge import BridgeResult, estimate as _estimate

# ── Overrun detection ─────────────────────────────────────────────────────────

def detect_overrun_risk(
    pm_input: PMInput,
    result: BridgeResult,
    budget_pm: float,
) -> dict:
    """Assess cost overrun risk given a budget in person-months.

    Returns:
        risk_level   : 'low' | 'medium' | 'high'
        overrun_pm   : estimated overrun in person-months (0 if within budget)
        overrun_pct  : overrun as % of budget
        message      : human-readable risk summary
    """
    ratio = result.effort_pm / max(budget_pm, 0.01)
    overrun_pm = max(0.0, result.effort_pm - budget_pm)
    overrun_pct = round(overrun_pm / max(budget_pm, 0.01) * 100.0, 1)

    if ratio <= 0.85:
        risk_level = "low"
        message = f"Estimate ({result.effort_pm:.1f} pm) is well within budget ({budget_pm:.1f} pm). {round((1-ratio)*100)}% headroom."
    elif ratio <= 1.0:
        risk_level = "medium"
        message = f"Estimate ({result.effort_pm:.1f} pm) is close to budget ({budget_pm:.1f} pm). Little margin for unexpected work."
    else:
        risk_level = "high"
        message = f"Estimate ({result.effort_pm:.1f} pm) exceeds budget ({budget_pm:.1f} pm) by {overrun_pm:.1f} pm ({overrun_pct}%). Action required."

    return {
        "risk_level": risk_level,
        "overrun_pm": round(overrun_pm, 2),
        "overrun_pct": overrun_pct,
        "message": message,
    }


# ── Resource optimization ─────────────────────────────────────────────────────

# Levers that can reduce effort, with direction and description template
_OPTIMIZATION_LEVERS = [
    # (field, reduction_direction, label_fn)
    ("feature_complexity", "reduce",  lambda v: {"low": "medium", "medium": "low", "high": "medium"}.get(v, v)),
    ("num_features",       "reduce",  lambda v: max(1, round(v * 0.8))),
    ("uncertainty_level",  "reduce",  lambda v: {"high": "medium", "medium": "low"}.get(v, v)),
    ("technical_debt",     "reduce",  lambda v: {"high": "low", "low": "none"}.get(v, v)),
    ("requirement_change_risk", "reduce", lambda v: {"high": "medium", "medium": "low"}.get(v, v)),
    ("seniority",          "increase", lambda v: {"junior": "mid", "mid": "senior"}.get(v, v)),
    ("stack_experience",   "increase", lambda v: {"low": "medium", "medium": "high"}.get(v, v)),
    ("num_apis",           "reduce",  lambda v: max(0, round(v * 0.8))),
]


def suggest_optimizations(
    pm_input: PMInput,
    result: BridgeResult,
    target_pm: float,
) -> list[dict]:
    """Return sorted list of optimization suggestions to reduce effort to target_pm.

    Each suggestion has: field, current_value, suggested_value,
                         estimated_saving_pm, description
    """
    if result.effort_pm <= target_pm:
        return []

    suggestions = []
    for field, direction, transform in _OPTIMIZATION_LEVERS:
        current_val = getattr(pm_input, field)
        new_val = transform(current_val)
        if new_val == current_val:
            continue

        try:
            sim = _estimate(_with_override(pm_input, field, new_val))
        except Exception:
            continue

        if sim is None:
            continue

        saving = result.effort_pm - sim.effort_pm
        if saving <= 0:
            continue

        action = "Reduce" if direction == "reduce" else "Increase"
        suggestions.append({
            "field": field,
            "current_value": current_val,
            "suggested_value": new_val,
            "estimated_saving_pm": round(saving, 2),
            "description": f"{action} '{field}' from '{current_val}' to '{new_val}' -> saves ~{saving:.1f} pm",
        })

    return sorted(suggestions, key=lambda s: s["estimated_saving_pm"], reverse=True)


def _with_override(pm_input: PMInput, field: str, value) -> PMInput:
    overridden = copy.copy(pm_input)
    object.__setattr__(overridden, field, value)
    overridden._validate()
    return overridden


# ── Risk advisory ─────────────────────────────────────────────────────────────

_RISK_ADVICE = {
    "uncertainty_level": {
        "high": "Run a discovery/spike phase before committing to the full scope. Define acceptance criteria early.",
        "medium": "Hold a risk review at each sprint. Document unknowns and assign owners.",
    },
    "technical_debt": {
        "high": "Allocate 20-30% of capacity for debt reduction. Unaddressed debt will compound delivery risk.",
        "low": "Schedule periodic refactoring sessions to prevent debt from growing.",
    },
    "requirement_change_risk": {
        "high": "Adopt iterative delivery with short sprints. Freeze requirements per sprint.",
        "medium": "Establish a change control process: all requirement changes require PM sign-off.",
    },
    "feature_complexity": {
        "high": "Break complex features into sub-tasks before estimation. Consider technical spikes.",
    },
    "security_level": {
        "high": "Involve a security engineer from day one. Budget for penetration testing.",
    },
    "budget_constraint": {
        "hard": "Prioritize a MVP scope. Define must-have vs nice-to-have features explicitly.",
    },
    "seniority": {
        "junior": "Pair junior developers with a senior mentor. Add 15-20% buffer for ramp-up time.",
    },
    "stack_experience": {
        "low": "Budget for training and exploration time. Consider a tech-stack spike in week 1-2.",
    },
}


def advise_risks(pm_input: PMInput) -> list[dict]:
    """Return mitigation recommendations for high-risk PM input fields.

    Each recommendation: field, value, advice
    """
    advice_list = []
    fields = {f: getattr(pm_input, f) for f in _RISK_ADVICE}

    for field, value in fields.items():
        field_advice = _RISK_ADVICE.get(field, {})
        text = field_advice.get(value)
        if text:
            advice_list.append({"field": field, "value": value, "advice": text})

    return advice_list
