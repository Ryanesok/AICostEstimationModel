## MODIFIED Requirements

### Requirement: Chart renders bars for all six models with selected model highlighted
The right-panel chart SHALL render one horizontal bar per model (6 bars total); the bar for the currently selected model SHALL use a distinct highlight color; all other bars SHALL use a muted color.

#### Scenario: All six model bars appear after estimation
- **WHEN** the user clicks Estimasi successfully
- **THEN** the chart renders 6 horizontal bars, one per model (SVR, LR, RF, XGBoost, LSTM, Hybrid)
- **THEN** bars are ordered top-to-bottom by model name (consistent order regardless of selection)
- **THEN** each bar shows the predicted person-months value as a label

#### Scenario: Selected model bar is visually highlighted
- **WHEN** the chart is rendered after estimation
- **THEN** the bar whose label matches the currently selected model SHALL use the accent color (`#4a9eff`)
- **THEN** all other bars SHALL use the muted color (`#555555`)

#### Scenario: Chart highlight updates when model selection changes
- **WHEN** the user changes the model selection in the dropdown after a successful estimation
- **THEN** the chart SHALL re-render with the new model's bar highlighted
- **THEN** no new estimation is required to update the highlight

#### Scenario: Empty chart before estimation
- **WHEN** the application loads or after Reset Form
- **THEN** the chart shows the empty placeholder text (unchanged from current behavior)
