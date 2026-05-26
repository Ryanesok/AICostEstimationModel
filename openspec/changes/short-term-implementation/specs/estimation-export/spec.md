## ADDED Requirements

### Requirement: Export button in results panel
The GUI SHALL display an "Export" button in the results panel that becomes active after at least one estimation has been run. Clicking it SHALL open a native save-file dialog pre-filtered to `.csv` and `.json` extensions.

#### Scenario: Export button enabled after estimation
- **WHEN** the user runs an estimation and predictions are displayed
- **THEN** the Export button SHALL become enabled (clickable)

#### Scenario: Export button disabled before estimation
- **WHEN** the results panel is shown but no estimation has been run yet
- **THEN** the Export button SHALL be disabled (non-clickable, visually dimmed)

### Requirement: CSV export format
When the user selects a `.csv` destination, the system SHALL write a single-row CSV file where each column is either an input field (column name = field key, value = user input) or a model prediction (column name = `pred_<model_name>`, value = the numeric prediction). A `timestamp` column with ISO-8601 format SHALL be included.

#### Scenario: CSV file content
- **WHEN** the user saves with a `.csv` extension
- **THEN** the file SHALL contain one header row and one data row with all input fields, all model predictions, and a timestamp

### Requirement: JSON export format
When the user selects a `.json` destination, the system SHALL write a JSON object with three keys: `"inputs"` (dict of field key → value), `"predictions"` (dict of model name → numeric prediction), and `"timestamp"` (ISO-8601 string).

#### Scenario: JSON file content
- **WHEN** the user saves with a `.json` extension
- **THEN** the file SHALL be valid JSON matching the three-key schema described above

### Requirement: Export error handling
If the file cannot be written (e.g., permission denied, disk full), the system SHALL display an error message in the GUI and SHALL NOT crash.

#### Scenario: Write failure
- **WHEN** the export operation fails due to an OS-level error
- **THEN** an error message SHALL appear in the GUI and the application SHALL remain usable
