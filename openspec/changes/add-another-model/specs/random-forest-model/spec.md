## ADDED Requirements

### Requirement: Random Forest is trained and exported per dataset
The training pipeline SHALL train a `RandomForestRegressor(n_estimators=200, random_state=42)` for each dataset and serialize it to `models/rf_<stem>.pkl`.

#### Scenario: RF model trained alongside SVR and LR
- **WHEN** `train_and_export()` runs for any dataset
- **THEN** a Random Forest model is trained on the full preprocessed data (same `X`, `y`, `scaler` as SVR/LR)
- **THEN** the fitted model is saved to `models/rf_<stem>.pkl`

#### Scenario: RF CV metrics included in metrics JSON
- **WHEN** training completes for a dataset
- **THEN** `models/<stem>_metrics.json` SHALL include `rf_cv_mae_mean` and `rf_cv_r2_mean`

#### Scenario: Stale RF pkl removed by cleanup
- **WHEN** `cleanup_stale_models()` runs and a stem is no longer in the trained set
- **THEN** `models/rf_<stem>.pkl` is deleted alongside the other stale pkl files
