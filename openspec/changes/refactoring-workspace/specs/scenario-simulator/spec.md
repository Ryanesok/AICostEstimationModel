## ADDED Requirements

### Requirement: Single-variable scenario recalculation
`core/estimator_bridge.py` SHALL expose a `simulate(base_input, field, value)` function that returns a new `EstimatorResult` with one `PMInput` field overridden.

#### Scenario: Simulate with more developers
- **WHEN** `simulate(base_input, "num_developers", base_input.num_developers + 2)` is called
- **THEN** a new `EstimatorResult` is returned reflecting the updated team size

#### Scenario: Simulated result differs from base when relevant field changes
- **WHEN** `feature_complexity` is changed from "low" to "high"
- **THEN** the simulated effort estimate is higher than the base estimate

#### Scenario: Unchanged fields preserved in simulation
- **WHEN** only `deadline_months` is changed in a simulation
- **THEN** all other `PMInput` fields in the simulated input match the base input exactly

### Requirement: Simulation comparison output
`core/estimator_bridge.py` SHALL expose a `compare(base_result, simulated_result)` function that returns a dict with `delta_effort`, `delta_pct`, and `direction` (increase/decrease/unchanged).

#### Scenario: Positive delta for larger scope
- **WHEN** simulated effort > base effort
- **THEN** `direction` is "increase" and `delta_effort` > 0

#### Scenario: Unchanged returns direction unchanged
- **WHEN** a field is simulated to its current value (no actual change)
- **THEN** `direction` is "unchanged" and `delta_pct` is 0.0

### Requirement: Simulation UI panel scaffold
`ui/simulation.py` SHALL provide a panel widget that allows the PM to pick one parameter and a new value, then displays the recalculated result alongside the original.

#### Scenario: Panel renders without error
- **WHEN** `SimulationPanel(parent, base_input, base_result)` is instantiated
- **THEN** the widget is created without raising an exception

#### Scenario: Recalculate button triggers simulation
- **WHEN** the PM changes a field value and clicks "Recalculate"
- **THEN** the displayed effort estimate updates to the simulated result
