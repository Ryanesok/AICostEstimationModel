## Why

The current `model_pipeline.py` and `auto_configure.py` were built for speed-to-result: single train/test split, default hyperparameters, no validation of feature quality, and metrics that are printed but never saved. On small datasets (COCOMO-81: ~63 rows, Desharnais: ~77 rows), this produces unreliable model performance estimates and risks silent data leakage — harming correctness of predictions and blocking confident future development.

## What Changes

- **Cross-validation in training**: Replace single `train_test_split(test_size=0.2)` with k-fold cross-validation (k=5) to produce stable MAE/R² estimates on small datasets; keep a held-out test set for final reporting
- **SVR hyperparameter tuning**: Replace hardcoded `SVR(kernel="rbf")` with a `GridSearchCV` over `C`, `epsilon`, and `kernel` to find parameters that actually fit each dataset
- **Feature leakage guard in `auto_configure.py`**: Warn the user (and require confirmation) when a candidate feature is highly correlated (|r| > 0.95) with the selected target — indicates near-duplicate or derived columns
- **Feature correlation filter**: Optionally drop one of any pair of features with |r| > 0.90 to reduce multicollinearity before training
- **Evaluation export**: Save per-dataset metrics (CV MAE, CV R², test MAE, test R², best SVR params) to `models/<stem>_metrics.json` alongside the `.pkl` files
- **Data quality checks**: Validate minimum row count before splitting (raise clear error if < 20 rows); detect and report constant/near-constant columns in `auto_configure.py`

## Capabilities

### New Capabilities

- `cross-validated-training`: Training pipeline uses k-fold CV + final test evaluation, saves metrics JSON per dataset
- `svr-hyperparameter-tuning`: SVR trained via GridSearchCV; best params serialized into metrics JSON
- `feature-leakage-guard`: `auto_configure.py` computes target correlation for each candidate feature and warns before saving config

### Modified Capabilities

- none

## Impact

- `model_pipeline.py`: `preprocess()`, `train_models()`, `evaluate()`, `train_and_export()` — logic changes, public signatures may change slightly
- `auto_configure.py`: `configure_one()`, `collect_labels_hints()` — adds correlation check step after target selection
- `models/` directory: new `<stem>_metrics.json` files created alongside existing `.pkl` files
- `app.py`: no changes required — model loading and prediction API (`svr.predict`, `lr.predict`) unchanged
