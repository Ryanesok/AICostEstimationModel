## ADDED Requirements

### Requirement: XGBoost is trained via GridSearchCV and exported per dataset
The training pipeline SHALL train an `XGBRegressor` using `GridSearchCV` over a fixed parameter grid and serialize the best estimator to `models/xgb_<stem>.pkl`.

#### Scenario: XGBoost trained with hyperparameter search
- **WHEN** `train_and_export()` runs for any dataset
- **THEN** `GridSearchCV` searches over `n_estimators ∈ [100, 300]`, `max_depth ∈ [3, 5]`, `learning_rate ∈ [0.05, 0.1]`
- **THEN** the best estimator is fit on the full data and saved to `models/xgb_<stem>.pkl`
- **THEN** `best_xgb_params` is stored in `models/<stem>_metrics.json`

#### Scenario: XGBoost CV metrics included in metrics JSON
- **WHEN** training completes for a dataset
- **THEN** `models/<stem>_metrics.json` SHALL include `xgb_cv_mae_mean` and `xgb_cv_r2_mean`

#### Scenario: xgboost package declared in requirements
- **WHEN** the project dependencies are listed
- **THEN** `requirements.txt` SHALL include `xgboost`
