## ADDED Requirements

### Requirement: Save last inputs on estimation
After a successful estimation, the system SHALL write the current form values to `last_inputs/<dataset_name>.json` as a flat dict mapping each field key to its string value. The `last_inputs/` directory SHALL be created automatically if it does not exist.

#### Scenario: File written after estimation
- **WHEN** the user runs an estimation successfully
- **THEN** `last_inputs/<dataset_name>.json` SHALL exist and contain the field values that were submitted

#### Scenario: File overwritten on subsequent run
- **WHEN** the user runs a second estimation with different values
- **THEN** the JSON file SHALL contain the latest values, not the previous ones

### Requirement: Load last inputs on dataset selection
When the user selects a dataset from the dropdown, the system SHALL check for a corresponding `last_inputs/<dataset_name>.json` file. If found, each form field SHALL be pre-populated with the saved value. If no file exists, fields SHALL use their normal defaults.

#### Scenario: Fields pre-populated from saved state
- **WHEN** the user selects a dataset that has a saved `last_inputs` file
- **THEN** each input field SHALL show the previously saved value

#### Scenario: No file — defaults used
- **WHEN** the user selects a dataset with no saved `last_inputs` file
- **THEN** form fields SHALL display their normal defaults as if no persistence existed

### Requirement: Corrupt or invalid persistence file handled gracefully
If `last_inputs/<dataset_name>.json` cannot be parsed (e.g., malformed JSON, missing keys), the system SHALL fall back to default field values without raising an error or displaying a crash dialog.

#### Scenario: Malformed JSON file
- **WHEN** the persistence file contains invalid JSON
- **THEN** the form SHALL load with default values and the error SHALL be silently swallowed (optionally logged to console)
