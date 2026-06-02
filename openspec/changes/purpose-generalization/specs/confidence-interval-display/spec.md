## ADDED Requirements

### Requirement: Confidence range is displayed alongside the effort estimate
The system SHALL display a confidence range in the format "± N person-months" immediately below or beside the effort estimate result. The range SHALL be derived from the auto-selected model's cross-validated MAE (`*_cv_mae_mean`) stored in the metrics JSON. The unit of the range SHALL match the effort unit of the selected dataset (converting person-hours to person-months where applicable).

#### Scenario: CV MAE is available in metrics JSON
- **WHEN** the auto-selected model has a `*_cv_mae_mean` value in its metrics JSON
- **THEN** the confidence range SHALL display as "± {cv_mae_mean:.1f} person-months" below the effort estimate

#### Scenario: Dataset effort unit is person-hours
- **WHEN** the auto-selected dataset uses `effort_unit: person-hours`
- **THEN** the CV MAE value SHALL be divided by 160 (HOURS_PER_MONTH) before displaying the confidence range, to match the person-months conversion applied to the estimate

#### Scenario: CV MAE key is absent from metrics JSON
- **WHEN** the metrics JSON does not contain a CV MAE key for the selected model
- **THEN** the confidence range SHALL not be shown (the line SHALL be hidden or omitted, not shown as "± — person-months")

### Requirement: Confidence range is visible in the result panel only after estimation
The confidence range SHALL be hidden (or shown as "—") when no estimation has been performed in the current session. It SHALL appear only after the user presses "Estimasi" and a valid result is produced.

#### Scenario: No estimate produced yet
- **WHEN** the app has just started and no estimate has been produced
- **THEN** the confidence range area SHALL be blank or show a placeholder "—"

#### Scenario: Estimate produced
- **WHEN** the user presses "Estimasi" and the estimation succeeds
- **THEN** the confidence range SHALL be populated with the correct ± value

#### Scenario: User presses Reset Form
- **WHEN** the user presses "Reset Form"
- **THEN** the confidence range SHALL revert to blank or "—"

### Requirement: Confidence range value is read-only and not user-editable
The confidence range display SHALL be a non-interactive label. The user SHALL NOT be able to modify the confidence value directly.

#### Scenario: User attempts to interact with the confidence range
- **WHEN** the user clicks on the confidence range label
- **THEN** no edit mode SHALL be triggered; the label SHALL remain static
