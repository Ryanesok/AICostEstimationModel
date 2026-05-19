## ADDED Requirements

### Requirement: Random Forest training uses GridSearchCV
`train_rf()` SHALL perform GridSearchCV over a predefined parameter grid using the same KFold object used by other models, and SHALL return the best estimator together with the best parameters found.

#### Scenario: train_rf receives kfold and returns best estimator
- **WHEN** `train_rf(X, y, kfold)` is called
- **THEN** it SHALL run GridSearchCV with `scoring="neg_mean_absolute_error"` and `n_jobs=-1`, and return `(best_estimator, best_params)` where `best_estimator` is a fitted `RandomForestRegressor`

#### Scenario: RF parameter grid covers n_estimators, max_depth, min_samples_split
- **WHEN** GridSearchCV is configured
- **THEN** the grid SHALL include at least two values for each of `n_estimators`, `max_depth`, and `min_samples_split`

#### Scenario: RF best_params persisted to metrics JSON
- **WHEN** `model_pipeline.py` completes training for a dataset
- **THEN** `models/<dataset>_metrics.json` SHALL contain a `rf_best_params` key with the winning hyperparameter values as a dict
