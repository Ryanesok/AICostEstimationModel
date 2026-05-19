## ADDED Requirements

### Requirement: Hybrid model prediction is a weighted average of LSTM, XGBoost, and Linear Regression
The app SHALL compute the Hybrid prediction at inference time by combining the predictions of LSTM, XGBoost, and LinearRegression using weights inversely proportional to each model's CV MAE, stored in `models/<stem>_metrics.json`.

#### Scenario: Hybrid weights computed during training
- **WHEN** `train_and_export()` completes for a dataset
- **THEN** `models/<stem>_metrics.json` SHALL include `hybrid_weights: {"lstm": w1, "xgb": w2, "lr": w3}` where `w_i = (1/max(mae_i, 1e-6)) / sum(1/max(mae_j, 1e-6))`

#### Scenario: Hybrid prediction computed in app at inference
- **WHEN** the user clicks Estimasi with "Hybrid" selected
- **THEN** `app.py` calls `predict()` on the LSTM, XGBoost, and LR models separately
- **THEN** the weighted average is computed using `hybrid_weights` from the metrics JSON
- **THEN** the resulting value is treated identically to any other model's prediction (unit conversion, display, chart update)

#### Scenario: Hybrid not available if component models are missing
- **WHEN** any of LSTM, XGBoost, or LR pkl/pt files are missing for the selected dataset
- **THEN** the "Hybrid" option SHALL be disabled or produce a clear error message
