## ADDED Requirements

### Requirement: Training uses k-fold cross-validation for performance estimation
The training pipeline SHALL use `KFold(n_splits=5, shuffle=True, random_state=42)` to evaluate model performance; it SHALL NOT rely on a single train/test split as the sole performance metric.

#### Scenario: CV metrics are computed for each dataset
- **WHEN** `train_and_export()` is called for a dataset with at least 10 rows
- **THEN** cross-validated MAE mean and std are computed for both SVR and LinearRegression
- **THEN** cross-validated R² mean and std are computed for both models

#### Scenario: Final production model trained on full dataset
- **WHEN** CV evaluation completes successfully
- **THEN** the final SVR and LinearRegression models are fit on 100% of the preprocessed data
- **THEN** the fitted models are saved as `models/svr_<stem>.pkl` and `models/linreg_<stem>.pkl`
- **THEN** the fitted scaler is saved as `models/scaler_<stem>.pkl`

#### Scenario: Metrics persisted to JSON
- **WHEN** training and evaluation complete for a dataset
- **THEN** a metrics file is written to `models/<stem>_metrics.json` containing: `cv_mae_mean`, `cv_mae_std`, `cv_r2_mean`, `cv_r2_std`, `lr_cv_mae_mean`, `lr_cv_r2_mean`, `trained_on_rows`
- **THEN** the same metrics are printed to stdout in a readable format

#### Scenario: Minimum row guard
- **WHEN** a dataset has fewer than 10 usable rows after dropping NaN
- **THEN** training SHALL be skipped with a clear warning message
- **THEN** no pkl or metrics files SHALL be written for that dataset
