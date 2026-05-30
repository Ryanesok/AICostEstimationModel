## ADDED Requirements

### Requirement: Cost overrun risk detection
`core/decision_support.py` SHALL expose `detect_overrun_risk(pm_input: PMInput, result: BridgeResult, budget_pm: float) -> dict` that assesses the probability of cost overrun given a budget in person-months.

#### Scenario: Overrun detected when estimate exceeds budget
- **WHEN** `result.effort_pm > budget_pm`
- **THEN** `risk_level` is `"high"` and `overrun_pm` is positive

#### Scenario: Safe when estimate well within budget
- **WHEN** `result.effort_pm <= budget_pm * 0.85`
- **THEN** `risk_level` is `"low"`

#### Scenario: Warning zone when estimate near budget
- **WHEN** `0.85 < result.effort_pm / budget_pm <= 1.0`
- **THEN** `risk_level` is `"medium"`

### Requirement: Resource optimization suggestions
`core/decision_support.py` SHALL expose `suggest_optimizations(pm_input: PMInput, result: BridgeResult, target_pm: float) -> list[dict]` that returns actionable suggestions to reduce effort to `target_pm`.

#### Scenario: Suggestions returned when over target
- **WHEN** `result.effort_pm > target_pm`
- **THEN** at least one suggestion dict is returned with keys `field`, `current_value`, `suggested_value`, `estimated_saving_pm`, `description`

#### Scenario: Suggestions prioritized by savings
- **WHEN** multiple suggestions are available
- **THEN** the list is sorted by `estimated_saving_pm` descending

#### Scenario: Empty list when already within target
- **WHEN** `result.effort_pm <= target_pm`
- **THEN** an empty list is returned

### Requirement: Risk advisory recommendations
`core/decision_support.py` SHALL expose `advise_risks(pm_input: PMInput) -> list[dict]` that returns mitigation recommendations for each high-risk PM field.

#### Scenario: Recommendation for high uncertainty
- **WHEN** `pm_input.uncertainty_level == "high"`
- **THEN** a recommendation dict is returned referencing uncertainty with a concrete mitigation action

#### Scenario: No recommendations for low-risk input
- **WHEN** all risk fields are "low" or "none"
- **THEN** an empty list is returned

### Requirement: Decision support exposed from core
`core/__init__.py` SHALL expose `decision_support` module.

#### Scenario: decision_support importable from core
- **WHEN** `from core import decision_support` is executed
- **THEN** no `ImportError` is raised
