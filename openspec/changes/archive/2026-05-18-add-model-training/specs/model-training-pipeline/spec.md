## REMOVED Requirements

### Requirement: Load or generate training data
**Reason**: Replaced by multi-dataset batch pipeline that reads all CSVs from `data/` using `DATASET_CONFIG`; the single-file dummy-data fallback is superseded.
**Migration**: Run `downloader.py` to populate `data/`, then `model_pipeline.py` for batch training.

### Requirement: Export trained artifacts
**Reason**: Artifact naming convention changes from `svr_model.pkl` / `lr_model.pkl` / `scaler.pkl` to `svr_<stem>.pkl` / `linreg_<stem>.pkl` / `scaler_<stem>.pkl`.
**Migration**: Delete old `.pkl` files and re-run `model_pipeline.py` after upgrading.

## MODIFIED Requirements

### Requirement: Train SVR and LinearRegression models
The pipeline SHALL train both `sklearn.svm.SVR` (RBF kernel) and `sklearn.linear_model.LinearRegression` for **each** CSV found in `data/` that has a matching entry in `DATASET_CONFIG`. Training uses an 80/20 split with `random_state=42`.

#### Scenario: Both models trained per dataset
- **WHEN** the pipeline runs and finds N datasets with valid config
- **THEN** 2N model files plus N scaler files are created in `models/`

#### Scenario: Reproducible split
- **WHEN** the pipeline is run multiple times on the same data
- **THEN** `random_state=42` ensures the same train/test split each time

### Requirement: Evaluate and report model performance
After training each dataset, the pipeline SHALL compute and print MAE and R2 for both models on the held-out test set, prefixed by the dataset name.

#### Scenario: Metrics printed per dataset to stdout
- **WHEN** training and evaluation complete for one dataset
- **THEN** stdout shows `[<dataset>] SVR MAE=X R2=Y` and `[<dataset>] LinearRegression MAE=X R2=Y`
