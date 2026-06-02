## ADDED Requirements

### Requirement: Effort to currency conversion
`core/cost_calculator.py` SHALL expose a `compute(effort_pm, salary_per_month, deadline_months, num_developers)` function that returns total project cost and team sizing recommendations.

#### Scenario: Total cost computed correctly
- **WHEN** `compute(effort_pm=18, salary_per_month=15_000_000, deadline_months=6, num_developers=4)` is called
- **THEN** `total_cost` equals `18 * 15_000_000 = 270_000_000`

#### Scenario: Team needed computed correctly
- **WHEN** `effort_pm=18` and `deadline_months=6`
- **THEN** `team_needed = ceil(18 / 6) = 3`

### Requirement: Cost result dataclass
`cost_calculator.py` SHALL return a `CostResult` dataclass with fields: `total_cost`, `team_needed`, `cost_per_developer`, `currency`, `formatted_total`.

#### Scenario: Formatted total uses IDR locale
- **WHEN** `currency="IDR"` and `total_cost=270_000_000`
- **THEN** `formatted_total` is `"Rp 270,000,000"`

#### Scenario: cost_per_developer correct
- **WHEN** `total_cost=270_000_000` and `num_developers=3`
- **THEN** `cost_per_developer = 90_000_000`

### Requirement: Cost calculator exposed from core
`core/__init__.py` SHALL expose `cost_calculator` module and `CostResult` dataclass.

#### Scenario: CostResult importable from core
- **WHEN** `from core import CostResult` is executed
- **THEN** no `ImportError` is raised
