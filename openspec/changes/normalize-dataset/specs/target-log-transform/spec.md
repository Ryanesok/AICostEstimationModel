## ADDED Requirements

### Requirement: Dataset config supports log-transform flag
Each dataset entry in `dataset_config.yaml` MAY contain a boolean field `log_transform_target`. When `true`, the pipeline SHALL apply `np.log1p` to the target variable before training and `np.expm1` on all predictions at inference time. When absent or `false`, behavior SHALL be identical to the current implementation (no transform applied).

#### Scenario: Flag present and true
- **WHEN** a dataset entry has `log_transform_target: true`
- **THEN** `preprocess()` returns `y` as `np.log1p(raw_y)` and also returns the flag value `True`

#### Scenario: Flag absent or false
- **WHEN** a dataset entry has `log_transform_target: false` or the key is missing
- **THEN** `preprocess()` returns raw `y` unchanged and returns flag value `False`

---

### Requirement: Metrics computed in original scale
When `log_transform_target` is `true`, the pipeline SHALL inverse-transform (`np.expm1`) OOF predictions and true values before computing MAE, RMSE, MMRE, and PRED25. All metric values stored in `_metrics.json` MUST be in the original effort unit (person-hours or person-months).

#### Scenario: CV metrics with log-transform active
- **WHEN** log-transform is active and CV fold produces predictions in log-space
- **THEN** apply `np.expm1` to both `y_pred` and `y_true` before computing MAE, RMSE, MMRE, PRED25

#### Scenario: CV metrics without log-transform
- **WHEN** log-transform is not active
- **THEN** metrics are computed directly from raw predictions (existing behavior, unchanged)

---

### Requirement: Log-transform flag persisted to metrics JSON
`evaluate_models()` SHALL write `log_transform_target: true/false` into the `_metrics.json` dict for the dataset, so that inference-time code can read the flag without re-reading `dataset_config.yaml`.

#### Scenario: Metrics file written after training
- **WHEN** `evaluate_models()` completes for a dataset with `log_transform_target: true`
- **THEN** the resulting `_metrics.json` contains `"log_transform_target": true`

#### Scenario: Legacy metrics file without flag
- **WHEN** reading a `_metrics.json` that does not contain `log_transform_target`
- **THEN** code defaults to `false` via `.get("log_transform_target", False)` — no `KeyError`

---

### Requirement: Inference applies inverse-transform when flag is active
`run_estimate()` and `run_hybrid_estimate()` in `estimator.py` SHALL apply `np.expm1` to the raw model output if the loaded model was trained with `log_transform_target: true`. The flag MUST be loaded from the dataset's `_metrics.json` at model-load time and stored on `LoadedModels`.

#### Scenario: Prediction with log-transform active
- **WHEN** `run_estimate()` is called and `LoadedModels.log_transform_target` is `True`
- **THEN** final prediction = `np.expm1(raw_model_output)` before unit conversion

#### Scenario: Prediction without log-transform
- **WHEN** `LoadedModels.log_transform_target` is `False` or absent
- **THEN** raw model output is used directly (existing behavior)

---

### Requirement: auto_configure.py detects skewness and sets flag
When `auto_configure.py` processes a new dataset, it SHALL compute the skewness of the target column. If `abs(skewness) > 1.0`, it SHALL set `log_transform_target: true` for that dataset in `dataset_config.yaml`; otherwise `false`. For datasets already present in config, the flag SHALL NOT be overwritten automatically.

#### Scenario: High-skew target on new dataset
- **WHEN** a new dataset is configured and its target column has skewness > 1.0
- **THEN** `dataset_config.yaml` is written with `log_transform_target: true` for that dataset

#### Scenario: Low-skew target on new dataset
- **WHEN** a new dataset is configured and its target column has skewness ≤ 1.0
- **THEN** `dataset_config.yaml` is written with `log_transform_target: false` for that dataset

#### Scenario: Existing dataset in config
- **WHEN** `auto_configure.py` runs on a dataset already present in `dataset_config.yaml`
- **THEN** the existing `log_transform_target` value is preserved unchanged
