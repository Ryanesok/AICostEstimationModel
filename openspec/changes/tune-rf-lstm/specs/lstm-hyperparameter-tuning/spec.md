## ADDED Requirements

### Requirement: LSTM training performs grid search over hidden_size and lr
`train_lstm()` SHALL accept optional grid-search mode where it iterates over candidate `hidden_size` and `lr` values using cross-validation MAE as the selection criterion, then retrains on the full dataset using the winning parameters.

#### Scenario: LSTM grid search selects params with lowest CV MAE
- **WHEN** `train_lstm_with_tuning(X, y, kfold)` is called
- **THEN** it SHALL evaluate all combinations of `hidden_size` ∈ {32, 64, 128} and `lr` ∈ {1e-3, 5e-4} via 5-fold CV, select the combination with the lowest mean MAE, and return `(model, best_params)`

#### Scenario: Seed is set before each LSTM candidate training run
- **WHEN** each candidate combination is evaluated during grid search
- **THEN** `torch.manual_seed` SHALL be called with a fixed seed before instantiating the model, ensuring reproducible fold results

#### Scenario: LSTM best_params persisted to metrics JSON
- **WHEN** `model_pipeline.py` completes training for a dataset
- **THEN** `models/<dataset>_metrics.json` SHALL contain an `lstm_best_params` key with `hidden_size` and `lr` values as a dict

#### Scenario: Final LSTM model retrained on full data with best params
- **WHEN** the best `(hidden_size, lr)` combination is identified
- **THEN** a new `LSTMModel` SHALL be trained on the full training dataset (not a CV fold) using those params before being saved to disk
