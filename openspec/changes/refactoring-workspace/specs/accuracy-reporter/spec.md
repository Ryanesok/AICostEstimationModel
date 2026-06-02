## ADDED Requirements

### Requirement: Model accuracy summary for PM
`core/accuracy_reporter.py` SHALL expose a `summarize(result: BridgeResult) -> dict` function that extracts and formats model quality metrics in PM-readable form.

#### Scenario: Summary returned for any BridgeResult
- **WHEN** `accuracy_reporter.summarize(result)` is called
- **THEN** a dict with keys `model_name`, `dataset`, `trained_on`, `mape_pct`, `mae_pm`, `pred25_pct`, `quality_label` is returned

#### Scenario: quality_label assigned correctly
- **WHEN** `mmre <= 0.10` (MAPE <= 10%)
- **THEN** `quality_label` is `"Excellent"`
- **WHEN** `0.10 < mmre <= 0.25`
- **THEN** `quality_label` is `"Good"`
- **WHEN** `mmre > 0.25`
- **THEN** `quality_label` is `"Fair"` and a warning flag is set

### Requirement: Low-quality model warning
`accuracy_reporter.py` SHALL set `warn=True` in the summary dict when model quality is below the SRS threshold (MAPE > 15%, i.e., mmre > 0.15), so the UI can display a warning to the PM.

#### Scenario: Warning triggered for poor model
- **WHEN** `mmre > 0.15`
- **THEN** `summary['warn']` is `True` and `summary['warn_message']` is non-empty

#### Scenario: No warning for good model
- **WHEN** `mmre <= 0.10`
- **THEN** `summary['warn']` is `False`

### Requirement: Accuracy reporter exposed from core
`core/__init__.py` SHALL expose `accuracy_reporter` module.

#### Scenario: accuracy_reporter importable from core
- **WHEN** `from core import accuracy_reporter` is executed
- **THEN** no `ImportError` is raised
