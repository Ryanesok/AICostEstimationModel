## ADDED Requirements

### Requirement: PM input parameter schema
The system SHALL define a typed `PMInput` dataclass in `core/schema.py` covering four parameter dimensions — Technical, Team, Business, and Risk — as specified in `project_parameter.txt`.

#### Scenario: PM input can be constructed with all parameters
- **WHEN** all 15 PM input fields are provided with valid values
- **THEN** a `PMInput` instance is created without error

#### Scenario: PM input serializes to dict
- **WHEN** `PMInput` instance is converted to a dict (e.g., `dataclasses.asdict`)
- **THEN** all fields are present with their values

### Requirement: Technical dimension fields
`PMInput` SHALL include: `num_features` (int), `feature_complexity` (str: low/medium/high), `num_apis` (int), `platform` (str), `third_party_integrations` (int), `security_level` (str: basic/standard/high).

#### Scenario: Invalid feature_complexity rejected at validation
- **WHEN** `feature_complexity` is set to a value outside `["low", "medium", "high"]`
- **THEN** a `ValueError` is raised with a message identifying the invalid field

#### Scenario: Valid technical fields accepted
- **WHEN** `num_features=10`, `feature_complexity="high"`, `num_apis=5`, `platform="web"`, `third_party_integrations=3`, `security_level="standard"` are provided
- **THEN** the Technical fields are stored correctly on the instance

### Requirement: Team dimension fields
`PMInput` SHALL include: `num_developers` (int ≥ 1), `seniority` (str: junior/mid/senior/mixed), `stack_experience` (str: low/medium/high).

#### Scenario: num_developers zero rejected
- **WHEN** `num_developers=0` is provided
- **THEN** a `ValueError` is raised

#### Scenario: Valid team fields accepted
- **WHEN** `num_developers=3`, `seniority="mid"`, `stack_experience="high"` are provided
- **THEN** the Team fields are stored correctly on the instance

### Requirement: Business dimension fields
`PMInput` SHALL include: `deadline_months` (float > 0), `budget_constraint` (str: none/soft/hard), `requirement_change_risk` (str: low/medium/high).

#### Scenario: deadline_months zero rejected
- **WHEN** `deadline_months=0.0` is provided
- **THEN** a `ValueError` is raised

#### Scenario: Valid business fields accepted
- **WHEN** `deadline_months=6.0`, `budget_constraint="soft"`, `requirement_change_risk="medium"` are provided
- **THEN** the Business fields are stored correctly on the instance

### Requirement: Risk dimension fields
`PMInput` SHALL include: `uncertainty_level` (str: low/medium/high), `technical_debt` (str: none/low/high), `external_dependencies` (int ≥ 0).

#### Scenario: Valid risk fields accepted
- **WHEN** `uncertainty_level="medium"`, `technical_debt="low"`, `external_dependencies=2` are provided
- **THEN** the Risk fields are stored correctly on the instance
