## ADDED Requirements

### Requirement: Column-mapping configuration
The pipeline SHALL maintain a `DATASET_CONFIG` dictionary keyed by CSV filename stem (lowercase). Each entry SHALL define `features` (list of column names to use as X) and `target` (column name to use as y).

#### Scenario: Known dataset processed
- **WHEN** a CSV filename stem matches a key in `DATASET_CONFIG`
- **THEN** the pipeline uses the configured features and target columns for that file

#### Scenario: Unknown dataset skipped
- **WHEN** a CSV filename stem has no entry in `DATASET_CONFIG`
- **THEN** the pipeline logs a WARNING and skips that file without crashing

---

### Requirement: Batch-train all CSVs in data/
The pipeline SHALL iterate every `.csv` file found in the `data/` directory and train one SVR and one LinearRegression model per file (using the column mapping).

#### Scenario: Multiple CSVs trained in one run
- **WHEN** `data/` contains two or more CSV files with known configs
- **THEN** the pipeline trains and exports a model pair for each file

#### Scenario: Empty data/ directory
- **WHEN** `data/` contains no CSV files
- **THEN** the pipeline prints a warning and exits cleanly without creating any `.pkl` files

---

### Requirement: Per-dataset artifact naming
Exported model files SHALL follow the naming pattern `models/svr_<stem>.pkl`, `models/linreg_<stem>.pkl`, and `models/scaler_<stem>.pkl`, where `<stem>` is the lowercased CSV filename without extension.

#### Scenario: Artifacts named correctly
- **WHEN** training completes on `data/cocomo81.csv`
- **THEN** `models/svr_cocomo81.pkl`, `models/linreg_cocomo81.pkl`, and `models/scaler_cocomo81.pkl` are created

---

### Requirement: Preprocessing per dataset
For each dataset the pipeline SHALL drop rows with missing values, apply `StandardScaler` to the feature columns, and perform an 80/20 train/test split with `random_state=42`.

#### Scenario: Missing values removed
- **WHEN** a CSV contains rows with NaN in feature or target columns
- **THEN** those rows are dropped before training

---

### Requirement: Per-dataset evaluation output
After training each dataset, the pipeline SHALL print MAE and R2 for both SVR and LinearRegression on the held-out test set.

#### Scenario: Metrics printed per dataset
- **WHEN** training and evaluation complete for one dataset
- **THEN** stdout shows dataset name, model name, MAE, and R2 before moving to the next dataset
