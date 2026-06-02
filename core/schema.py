from __future__ import annotations
from dataclasses import dataclass

# Allowed values for categorical PM fields
_COMPLEXITY = {"low", "medium", "high"}
_SENIORITY = {"junior", "mid", "senior", "mixed"}
_STACK_EXP = {"low", "medium", "high"}
_BUDGET_CONSTRAINT = {"none", "soft", "hard"}
_CHANGE_RISK = {"low", "medium", "high"}
_UNCERTAINTY = {"low", "medium", "high"}
_TECH_DEBT = {"none", "low", "high"}
_SECURITY = {"basic", "standard", "high"}

_FIELD_ALLOWED: dict[str, set[str]] = {
    "feature_complexity": _COMPLEXITY,
    "seniority": _SENIORITY,
    "stack_experience": _STACK_EXP,
    "budget_constraint": _BUDGET_CONSTRAINT,
    "requirement_change_risk": _CHANGE_RISK,
    "uncertainty_level": _UNCERTAINTY,
    "technical_debt": _TECH_DEBT,
    "security_level": _SECURITY,
}


@dataclass
class PMInput:
    # ── Technical ─────────────────────────────────────────────────────────────
    num_features: int                   # total features / use-cases
    feature_complexity: str             # low | medium | high
    num_apis: int                       # internal + external API endpoints
    platform: str                       # web | mobile | desktop | embedded | other
    third_party_integrations: int       # count of 3rd-party services
    security_level: str                 # basic | standard | high

    # ── Team ──────────────────────────────────────────────────────────────────
    num_developers: int                 # number of developers (≥1)
    seniority: str                      # junior | mid | senior | mixed
    stack_experience: str               # low | medium | high

    # ── Business ──────────────────────────────────────────────────────────────
    deadline_months: float              # project duration in months (>0)
    budget_constraint: str              # none | soft | hard
    requirement_change_risk: str        # low | medium | high

    # ── Risk ──────────────────────────────────────────────────────────────────
    uncertainty_level: str             # low | medium | high
    technical_debt: str                # none | low | high
    external_dependencies: int         # count of external system dependencies (≥0)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for field_name, allowed in _FIELD_ALLOWED.items():
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(
                    f"'{field_name}' must be one of {sorted(allowed)}, got '{value}'"
                )
        if self.num_developers < 1:
            raise ValueError("'num_developers' must be >= 1")
        if self.deadline_months <= 0:
            raise ValueError("'deadline_months' must be > 0")
        if self.external_dependencies < 0:
            raise ValueError("'external_dependencies' must be >= 0")
        if self.num_features < 1:
            raise ValueError("'num_features' must be >= 1")
        if self.num_apis < 0:
            raise ValueError("'num_apis' must be >= 0")
