## ADDED Requirements

### Requirement: Random Forest hyperparameter search
The training pipeline SHALL perform `RandomizedSearchCV` over a defined parameter grid for the Random Forest model, using the same 5-fold cross-validation scorer (negative MAE) already used for SVR and XGBoost. The best parameter combination SHALL be stored in `models/<dataset>_metrics.json` under the key `best_params_rf`.

#### Scenario: RF training records best params
- **WHEN** `model_pipeline.py` trains the Random Forest for a dataset
- **THEN** the resulting `_metrics.json` SHALL contain a `best_params_rf` dict with the winning `n_estimators`, `max_depth`, and `min_samples_split` values

#### Scenario: RF uses best params at inference time
- **WHEN** `app.py` loads the saved RF model and runs a prediction
- **THEN** the loaded model SHALL use the hyperparameters selected during tuning, not hardcoded defaults

### Requirement: LSTM hyperparameter grid search
The training pipeline SHALL iterate over a fixed grid of LSTM hyperparameter combinations (`hidden_size`, `num_layers`, `learning_rate`), train each configuration with 5-fold cross-validation, and select the configuration with the lowest mean MAE. The winning configuration SHALL be stored in `models/<dataset>_metrics.json` under the key `best_params_lstm` and used to retrain the final LSTM on the full training set.

#### Scenario: LSTM training records best params
- **WHEN** `model_pipeline.py` trains the LSTM for a dataset
- **THEN** the resulting `_metrics.json` SHALL contain a `best_params_lstm` dict with the selected `hidden_size`, `num_layers`, and `learning_rate`

#### Scenario: Grid exhausts all combinations
- **WHEN** LSTM hyperparameter search completes
- **THEN** every combination in the defined grid SHALL have been evaluated before the best is selected

### Requirement: Metrics backward compatibility
Existing `_metrics.json` files that predate this change SHALL remain loadable without errors. Code reading `best_params_rf` or `best_params_lstm` SHALL use `.get()` with a sensible default so old files do not cause `KeyError`.

#### Scenario: Old metrics file loaded
- **WHEN** a `_metrics.json` file without `best_params_rf` or `best_params_lstm` is loaded
- **THEN** the application SHALL not raise an error and SHALL fall back gracefully
