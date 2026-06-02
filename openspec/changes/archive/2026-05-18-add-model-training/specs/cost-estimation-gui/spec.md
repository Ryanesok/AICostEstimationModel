## REMOVED Requirements

### Requirement: Load serialized models on startup
**Reason**: Replaced by dynamic discovery — the GUI scans `models/svr_*.pkl` at startup instead of loading fixed filenames.
**Migration**: Re-run `model_pipeline.py` after upgrading; the GUI will auto-discover newly named models.

### Requirement: Model selection
**Reason**: The static SVR/LR toggle is now secondary to the dataset selector; model toggle stays but dataset selection is the primary control.
**Migration**: No user-facing change — the toggle still exists within the selected dataset context.

## MODIFIED Requirements

### Requirement: Project metrics input form
The GUI SHALL provide the same five labeled CTkEntry fields (KLOC, Reliability, Complexity, Team Size, Schedule) with hint text. The form SHALL be reset automatically when the user switches datasets from the dropdown.

#### Scenario: Valid numeric input accepted
- **WHEN** the user enters a numeric value in any input field
- **THEN** the value is accepted and retained until changed or reset

#### Scenario: Non-numeric input rejected
- **WHEN** the user enters a non-numeric value and presses "Estimasi"
- **THEN** the GUI displays an inline error identifying the invalid field and does NOT call the model

#### Scenario: Dataset switch resets form
- **WHEN** the user selects a different dataset from the dropdown
- **THEN** all input fields and prediction output are cleared

### Requirement: Instant prediction output
After the user clicks "Estimasi", the GUI SHALL scale inputs using the dataset-specific scaler, run inference with the selected model (SVR or LR), and display predicted effort (Person-Months) in the output area.

#### Scenario: Successful prediction with dataset-specific scaler
- **WHEN** all inputs are valid, a dataset is selected, and the user clicks "Estimasi"
- **THEN** the scaler for the selected dataset is used, and the predicted effort is displayed

### Requirement: Reset form
The GUI SHALL provide a "Reset Form" button that clears all input fields, the prediction output, and the calculator section.

#### Scenario: Reset clears all fields, output, and calculator
- **WHEN** the user clicks "Reset Form"
- **THEN** all CTkEntry fields, the result label, salary/duration inputs, and calculator outputs are cleared
