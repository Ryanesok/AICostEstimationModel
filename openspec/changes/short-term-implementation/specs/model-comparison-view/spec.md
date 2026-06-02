## ADDED Requirements

### Requirement: Multi-model selection checkboxes
The results panel SHALL display a checkbox for each available model (up to 5). After an estimation is run, the user SHALL be able to select any subset of models to include in the comparison view. At least one model MUST remain selected at all times.

#### Scenario: Default selection after estimation
- **WHEN** an estimation completes
- **THEN** all available models SHALL be checked by default

#### Scenario: Deselecting all models prevented
- **WHEN** the user attempts to uncheck the last remaining selected model
- **THEN** the checkbox SHALL remain checked and the action SHALL be ignored

### Requirement: Side-by-side comparison table
The results panel SHALL render a comparison table showing, for each selected model: the predicted effort value, and the three CV metrics (MAE, RMSE, R²) loaded from `_metrics.json`. Columns represent models; rows represent metrics.

#### Scenario: Table updates on checkbox change
- **WHEN** the user checks or unchecks a model checkbox
- **THEN** the comparison table SHALL update immediately to show only selected models

#### Scenario: Table shows correct metric values
- **WHEN** the comparison table is rendered
- **THEN** each cell SHALL display the value from `models/<dataset>_metrics.json` for the corresponding model and metric

### Requirement: Comparison bar chart
The results panel SHALL render a bar chart (using matplotlib, consistent with the existing chart) where each bar represents one selected model's predicted effort value. The chart SHALL update whenever the model selection changes.

#### Scenario: Bar chart reflects selection
- **WHEN** the user deselects a model
- **THEN** that model's bar SHALL be removed from the chart on the next render

#### Scenario: Chart labels
- **WHEN** the bar chart is rendered
- **THEN** each bar SHALL be labelled with the model name (truncated to 10 characters if necessary to prevent overflow)
