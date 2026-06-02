## ADDED Requirements

### Requirement: LSTM model is trained and exported per dataset
The training pipeline SHALL train a single-hidden-layer PyTorch LSTM treating each sample as a single-timestep sequence, and serialize the trained state to `models/lstm_<stem>.pt` and its architecture metadata to `models/lstm_arch_<stem>.json`.

#### Scenario: LSTM trained on full data with early stopping
- **WHEN** `train_and_export()` runs for any dataset
- **THEN** the LSTM is trained with Adam optimizer, MSE loss, up to 200 epochs, with early stopping on a 20% validation split (patience=20)
- **THEN** the model state dict is saved to `models/lstm_<stem>.pt`
- **THEN** `models/lstm_arch_<stem>.json` is saved with `{"input_size": <n_features>}`

#### Scenario: LSTM CV metrics included in metrics JSON
- **WHEN** training completes for a dataset
- **THEN** `models/<stem>_metrics.json` SHALL include `lstm_cv_mae_mean` and `lstm_cv_r2_mean`
- **THEN** these values are estimated via k-fold CV on the tabular data (same KFold as other models)

#### Scenario: LSTM loaded correctly at app startup
- **WHEN** `app.py` loads models for a dataset stem
- **THEN** the LSTM state dict is loaded from `lstm_<stem>.pt` using `lstm_arch_<stem>.json` to reconstruct the model architecture
- **THEN** the model is set to `eval()` mode before inference

#### Scenario: torch package declared in requirements
- **WHEN** the project dependencies are listed
- **THEN** `requirements.txt` SHALL include `torch`
