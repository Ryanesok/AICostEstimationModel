## ADDED Requirements

### Requirement: Past estimate persistence
`core/past_project.py` SHALL expose `save(pm_input: PMInput, result: BridgeResult, label: str = "")` that persists an estimation record to a local JSON store (`last_inputs/history.json`).

#### Scenario: Saved record retrievable
- **WHEN** `save(pm_input, result, label="ProjectA")` is called
- **THEN** the record appears in `load_all()` output

#### Scenario: Multiple records accumulate
- **WHEN** `save()` is called twice with different inputs
- **THEN** `load_all()` returns a list with both records

### Requirement: Similar past project lookup
`core/past_project.py` SHALL expose `find_similar(pm_input: PMInput, top_n: int = 3) -> list[dict]` that returns the top-N most similar past estimates using cosine similarity on the numeric PM fields.

#### Scenario: Similar projects returned for matching input
- **WHEN** a past project with identical fields exists in history
- **THEN** it appears as the first result with `similarity >= 0.99`

#### Scenario: Returns empty list when history is empty
- **WHEN** no past records exist
- **THEN** `find_similar()` returns an empty list

#### Scenario: top_n respected
- **WHEN** history has 10 records and `top_n=3`
- **THEN** exactly 3 records are returned

### Requirement: Past project record schema
Each saved record SHALL include: `timestamp`, `label`, `pm_input` (as dict), `effort_pm`, `model_name`, `dataset`, `confidence_pct`, `similarity` (set during lookup, None when saved).

#### Scenario: Record has all required fields after save
- **WHEN** a record is saved and retrieved via `load_all()`
- **THEN** all required fields are present in the record dict

### Requirement: Past project module exposed from core
`core/__init__.py` SHALL expose `past_project` module.

#### Scenario: past_project importable from core
- **WHEN** `from core import past_project` is executed
- **THEN** no `ImportError` is raised
