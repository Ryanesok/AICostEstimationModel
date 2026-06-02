## ADDED Requirements

### Requirement: Load or generate training data
The pipeline SHALL load a COCOMO-structured CSV from `data/dataset.csv` if it exists. If the file is absent, the pipeline SHALL generate synthetic COCOMO81-structured dummy data and emit a clearly visible WARNING to stdout indicating that real data is not in use.

#### Scenario: CSV file present
- **WHEN** `data/dataset.csv` exists and is readable
- **THEN** the pipeline loads it into a pandas DataFrame and proceeds with training

#### Scenario: CSV file absent
- **WHEN** `data/dataset.csv` does not exist
- **THEN** the pipeline generates synthetic dummy data, prints a WARNING, and proceeds with training using that dummy data

---

### Requirement: Preprocess features
The pipeline SHALL apply `StandardScaler` normalization to all numeric feature columns before training. The fitted scaler SHALL be serialized alongside the models so that the GUI applies identical scaling to user inputs.

#### Scenario: Scaler is fit and exported
- **WHEN** training completes
- **THEN** the fitted `StandardScaler` object is saved to `models/scaler.pkl`

---

### Requirement: Train SVR and LinearRegression models
The pipeline SHALL train both a `sklearn.svm.SVR` (with RBF kernel) and a `sklearn.linear_model.LinearRegression` model on the preprocessed training split (80% train / 20% test).

#### Scenario: Both models trained successfully
- **WHEN** the pipeline runs without error
- **THEN** `models/svr_model.pkl` and `models/lr_model.pkl` are created

#### Scenario: Reproducible split
- **WHEN** the pipeline is run multiple times on the same data
- **THEN** `random_state=42` ensures the same train/test split each time

---

### Requirement: Evaluate and report model performance
After training, the pipeline SHALL compute and print MAE (Mean Absolute Error) and R² (coefficient of determination) for both models on the held-out test set.

#### Scenario: Metrics printed to stdout
- **WHEN** training and evaluation complete
- **THEN** stdout shows MAE and R² for SVR and LinearRegression in a human-readable format

---

### Requirement: Export trained artifacts
The pipeline SHALL serialize the trained SVR model, LinearRegression model, and fitted scaler to the `models/` directory using `joblib.dump`. The `models/` directory SHALL be created automatically if it does not exist.

#### Scenario: models/ directory auto-created
- **WHEN** `models/` does not exist before pipeline runs
- **THEN** the pipeline creates it and writes all three `.pkl` files without error

#### Scenario: Artifacts overwrite on re-run
- **WHEN** the pipeline is run a second time
- **THEN** existing `.pkl` files are overwritten silently (no prompt required)
