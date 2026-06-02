## ADDED Requirements

### Requirement: Model selector supports all six models via a dropdown
The middle panel model selector SHALL use a `CTkOptionMenu` with options: SVR, Linear Regression, Random Forest, XGBoost, LSTM, Hybrid — replacing the two-option `CTkSegmentedButton`.

#### Scenario: All six models listed in selector
- **WHEN** the application opens and models are loaded for a dataset
- **THEN** the model selector SHALL show all six model names as options
- **THEN** "SVR" SHALL be selected by default

#### Scenario: Unavailable models are visually indicated
- **WHEN** a model's pkl/pt file does not exist for the current dataset
- **THEN** that model option SHALL remain in the list but produce a clear error message when selected and Estimasi is clicked
- **THEN** no crash SHALL occur

#### Scenario: Model selection persists across dataset switch
- **WHEN** the user changes the dataset
- **THEN** the previously selected model remains selected in the dropdown
- **THEN** if the selected model is unavailable for the new dataset, an error is shown on next Estimasi attempt
