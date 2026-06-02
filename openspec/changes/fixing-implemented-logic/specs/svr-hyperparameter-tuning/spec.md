## ADDED Requirements

### Requirement: SVR is trained via GridSearchCV with a fixed parameter grid
The training pipeline SHALL use `GridSearchCV` to select SVR hyperparameters; it SHALL NOT use hardcoded default `SVR()` parameters for the production model.

#### Scenario: Grid covers C, epsilon, and kernel
- **WHEN** SVR training begins for any dataset
- **THEN** `GridSearchCV` SHALL search over `C ∈ [0.1, 1, 10, 100]`, `epsilon ∈ [0.01, 0.1, 1.0]`, `kernel ∈ ['rbf', 'linear']`
- **THEN** the search SHALL use `scoring='neg_mean_absolute_error'` and the same KFold splitter used for CV evaluation

#### Scenario: Best estimator is saved to pkl
- **WHEN** `GridSearchCV.fit()` completes
- **THEN** `best_estimator_` is serialized to `models/svr_<stem>.pkl`
- **THEN** `best_params_` is included in `models/<stem>_metrics.json` under the key `best_svr_params`

#### Scenario: Best params reported to stdout
- **WHEN** training completes for a dataset
- **THEN** the best SVR parameters are printed alongside the CV metrics
