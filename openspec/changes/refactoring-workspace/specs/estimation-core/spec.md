## ADDED Requirements

### Requirement: Estimator bridge maps PM input to ML features
`core/estimator_bridge.py` SHALL translate a `PMInput` instance into the numeric feature vector expected by `pipeline.estimator.run_estimate()` and return an `EstimatorResult`.

#### Scenario: Bridge produces a result from valid PM input
- **WHEN** a fully-populated `PMInput` is passed to `estimator_bridge.estimate(pm_input)`
- **THEN** an `EstimatorResult` is returned with a non-zero effort value

#### Scenario: Bridge mapping is documented
- **WHEN** the mapping from each PM field to ML feature is read from `estimator_bridge.py`
- **THEN** every mapping assumption is described in an inline comment or docstring

### Requirement: Confidence interval calculation
`core/confidence.py` SHALL compute a confidence interval and a reliability score (0–100%) for a given `EstimatorResult`.

#### Scenario: Confidence interval returned for valid result
- **WHEN** `confidence.compute(estimator_result)` is called
- **THEN** a dict with keys `lower`, `upper`, `confidence_pct`, and `range_pct` is returned

#### Scenario: Confidence percentage is between 0 and 100
- **WHEN** any valid `EstimatorResult` is passed to `confidence.compute`
- **THEN** `confidence_pct` is a float in [0, 100]

### Requirement: Effort dominance breakdown
`core/dominance.py` SHALL compute the percentage share of effort attributable to each PM parameter dimension (Technical, Team, Business, Risk) for a given estimation result.

#### Scenario: Dominance shares sum to 100
- **WHEN** `dominance.compute(pm_input, estimator_result)` is called
- **THEN** the returned dict values sum to 100 (±0.1 for float rounding)

#### Scenario: Dominant dimension identified
- **WHEN** dominance is computed for a high-complexity, large-team input
- **THEN** one dimension has a higher share than the others

### Requirement: Package structure scaffold
The `core/` directory SHALL exist as a Python package with `__init__.py` exposing `schema`, `estimator_bridge`, `confidence`, and `dominance` modules.

#### Scenario: core package importable
- **WHEN** `import core` is executed from the project root
- **THEN** no `ImportError` is raised
