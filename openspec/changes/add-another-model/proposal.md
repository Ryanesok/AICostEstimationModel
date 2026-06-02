## Why

The app currently offers two models (SVR and Linear Regression). Both are limited: SVR can struggle with high-dimensional tabular data and LR is too simple for non-linear relationships. Adding ensemble and deep-learning-adjacent models gives users better predictive options and more confidence when selecting estimates, especially since datasets vary widely (63–499 rows, 5–25 features).

## What Changes

- **Random Forest** added as a standalone model: robust to outliers, naturally handles non-linear feature interactions, no hyperparameter tuning required for reasonable results
- **XGBoost** added as a standalone model: gradient boosting on tabular data, typically outperforms SVR and LR on small-to-medium datasets; requires `xgboost` package
- **LSTM** added as a standalone model: single-timestep PyTorch LSTM treating each row's features as one time step; provides a neural approach; requires `torch` package
- **Hybrid** added as a composite model: weighted-average ensemble of LSTM + XGBoost + Linear Regression, weights inversely proportional to each model's CV MAE; automatically adapts weights per dataset
- Model selector in the app upgraded from a 2-option `CTkSegmentedButton` to a `CTkOptionMenu` supporting all 6 models
- Chart updated to show bars for all 6 models simultaneously; the bar for the currently selected model is highlighted (distinct color), others rendered in muted tones
- `model_pipeline.py` extended to train, evaluate (CV), and export pkl for each new model
- `models/<stem>_metrics.json` extended with CV metrics for RF, XGBoost, LSTM, and Hybrid

## Capabilities

### New Capabilities

- `random-forest-model`: Random Forest trained per dataset, included in chart and model selector
- `xgboost-model`: XGBoost trained per dataset with CV tuning; `xgboost` added to requirements
- `lstm-model`: Single-timestep PyTorch LSTM trained per dataset; `torch` added to requirements
- `hybrid-model`: Weighted ensemble combining LSTM + XGBoost + LR predictions at inference time; no separate training required beyond component models

### Modified Capabilities

- `model-selector`: Selector upgrades from segmented button to dropdown; supports 6 model options
- `chart-visualization`: Chart now renders bars for all 6 models; selected model bar is highlighted

## Impact

- `model_pipeline.py`: add RF, XGBoost, LSTM training + export; extend metrics JSON; update cleanup
- `app.py`: replace `CTkSegmentedButton` with `CTkOptionMenu` for model selection; update `_on_estimate()` to dispatch to correct model; update chart to render 6 bars with highlight
- `requirements.txt`: add `xgboost`, `torch`
- `models/` directory: 3 new pkl files per dataset (`rf_`, `xgb_`, `lstm_`); hybrid computed at inference
- Existing `svr_` and `linreg_` pkl files and APIs unchanged
