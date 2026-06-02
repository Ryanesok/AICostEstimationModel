## ADDED Requirements

### Requirement: Best estimator is selected automatically at startup
The system SHALL read all `*_metrics.json` files in the `models/` directory on startup and select the dataset/model combination with the lowest cross-validated MAE without any user interaction. If two combinations are within 1% MAE of each other, the system SHALL use MMRE as a tiebreaker. The selected combination SHALL be applied to all subsequent estimations in the session.

#### Scenario: Single metrics file available
- **WHEN** only one `*_metrics.json` file exists in `models/`
- **THEN** the system SHALL load that dataset and use the model with the lowest `*_cv_mae_mean` key within it

#### Scenario: Multiple metrics files available
- **WHEN** multiple `*_metrics.json` files exist in `models/`
- **THEN** the system SHALL compare all dataset/model combinations and select the one with the globally lowest CV MAE

#### Scenario: Tie between two combinations within 1% MAE
- **WHEN** two combinations have CV MAE values within 1% of each other
- **THEN** the system SHALL select the one with the lower MMRE value, falling back to alphabetical dataset name if MMRE is also absent

#### Scenario: No metrics files found
- **WHEN** `models/` contains no `*_metrics.json` files
- **THEN** the system SHALL display the existing "Model Tidak Tersedia" error banner and disable the Estimasi button

### Requirement: Auto-selection is transparent via a technical detail section
The system SHALL expose which dataset and model were automatically selected in a collapsible UI section labelled "Detail Teknis", including the dataset name, model name, best hyperparameters, and the CV MAE score used for selection.

#### Scenario: User expands Detail Teknis
- **WHEN** the user clicks to expand "Detail Teknis"
- **THEN** the section SHALL display: selected dataset name, selected model name, CV MAE value, and best hyperparameter values from the metrics JSON

#### Scenario: Detail Teknis collapsed by default
- **WHEN** the application starts or an estimate is produced
- **THEN** the "Detail Teknis" section SHALL be collapsed by default

### Requirement: Input form matches the auto-selected dataset
The system SHALL render the input form fields corresponding to the auto-selected dataset's feature configuration from `dataset_config.yaml`. The form SHALL be rebuilt automatically when the auto-selected dataset changes (e.g., if the user re-runs training and restarts the app).

#### Scenario: Auto-selected dataset has a field config entry
- **WHEN** the auto-selected dataset stem has a matching entry in `dataset_config.yaml`
- **THEN** the input form SHALL display the dataset-specific labels, sliders, and hints for that dataset's features

#### Scenario: Auto-selected dataset has no field config entry
- **WHEN** the auto-selected dataset stem has no matching entry in `dataset_config.yaml`
- **THEN** the input form SHALL fall back to generic "Feature 1 … Feature N" labels based on the scaler's `n_features_in_`
