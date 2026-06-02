## ADDED Requirements

### Requirement: Result explanation narrative
`explainer/result_explainer.py` SHALL produce a human-readable explanation for an estimation result, describing why the estimate was generated and which factors had the most impact.

#### Scenario: Explanation produced for any result
- **WHEN** `result_explainer.explain(pm_input, estimator_result, dominance)` is called
- **THEN** a non-empty string is returned describing the estimate rationale

#### Scenario: Dominant factor mentioned in explanation
- **WHEN** the Technical dimension has the highest dominance share
- **THEN** the explanation string references technical complexity or features as a key driver

### Requirement: Reliability assessment
The explainer SHALL include a reliability statement based on the confidence score, categorizing it as: High (≥80%), Moderate (50–79%), or Low (<50%).

#### Scenario: High confidence labelled correctly
- **WHEN** `confidence_pct ≥ 80`
- **THEN** the explanation includes the word "High" or "reliable" in the reliability section

#### Scenario: Low confidence includes caveat
- **WHEN** `confidence_pct < 50`
- **THEN** the explanation includes a warning that the estimate has high uncertainty

### Requirement: High-cost factor callout
The explainer SHALL identify and surface the top 1–3 PM input factors that most increased the estimate, with a plain-language description.

#### Scenario: Cost drivers listed
- **WHEN** `explain` is called on an input with `feature_complexity="high"` and `uncertainty_level="high"`
- **THEN** the returned explanation lists at least one of those fields as a cost driver

### Requirement: Explainer package scaffold
The `explainer/` directory SHALL exist as a Python package with `__init__.py`.

#### Scenario: explainer package importable
- **WHEN** `import explainer` is executed from the project root
- **THEN** no `ImportError` is raised
