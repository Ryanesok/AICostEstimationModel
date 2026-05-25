## 1. Metrics Enhancement (model_pipeline.py)

- [x] 1.1 Add MMRE computation to `model_pipeline.py`: after CV predictions are collected for each model, compute `mean(|actual - predicted| / actual)` and store as `*_mmre_mean` in the metrics dict
- [x] 1.2 Add PRED(25) computation: compute fraction of CV predictions where relative error ≤ 0.25 and store as `*_pred25` in the metrics dict for each model (SVR, LR, RF, XGBoost, LSTM)
- [x] 1.3 Re-run `model_pipeline.py` for all datasets to regenerate `*_metrics.json` files with the new keys; verify keys appear in each JSON

## 2. Auto-Model Selection Module (estimator.py)

- [x] 2.1 Create `estimator.py` in the project root with a dataclass `EstimatorResult` holding: `stem`, `model_name`, `cv_mae`, `mmre`, `pred25`, `hyperparams`
- [x] 2.2 Implement `select_best_estimator(metrics_dir: str) -> EstimatorResult | None` that reads all `*_metrics.json`, scores by lowest `cv_mae_mean` (with MMRE tiebreaker within 1%), and returns the best result; return `None` if no files found
- [x] 2.3 Implement `load_estimator_models(stem: str, models_dir: str)` that loads SVR, LR, RF, XGBoost, LSTM, scaler, and hybrid weights for the selected stem (extract from current `_load_dataset` in `app.py`)
- [x] 2.4 Implement `run_estimate(model_name: str, X_scaled, X_tensor, models, unit: str) -> float | None` that runs a single model prediction and returns person-months (extract predict logic from current `_on_estimate`)

## 3. app.py Refactor — Auto-Selection Wiring

- [x] 3.1 On app startup, call `select_best_estimator(MODELS_DIR)` and store the result; if `None`, render the existing error banner and return early
- [x] 3.2 Remove the "Pilih Dataset" `CTkOptionMenu` from the left panel; remove the "Model Prediksi" `CTkOptionMenu` from the middle panel
- [x] 3.3 Load the auto-selected dataset's field config and models on startup via `load_estimator_models`; rebuild the input form for the selected stem
- [x] 3.4 Update `_on_estimate` to call `run_estimate` from `estimator.py` for the auto-selected model only (no loop over all models)

## 4. Trust Dashboard — Middle Panel Rebuild

- [x] 4.1 Replace the middle panel's model-picker dropdown with a result card showing: effort estimate (large text), unit label, and the ± confidence range label (initially "—")
- [x] 4.2 Add a quality label widget below the estimate: derive label text and color from the `EstimatorResult.mmre` threshold logic (≤0.25 green, ≤0.50 amber, >0.50 red, None gray)
- [x] 4.3 Add a compact 3-row metric table: MAE, MMRE, PRED(25) with values from `EstimatorResult` and one-line plain-language descriptions for each
- [x] 4.4 Add a collapsible "Detail Teknis" frame (using a toggle button + hidden frame pattern) that shows: dataset name, model name, CV MAE, MMRE, PRED(25), and best hyperparameters

## 5. Confidence Interval Display

- [x] 5.1 After estimation, compute confidence range: `cv_mae` from `EstimatorResult`, converted to person-months if `effort_unit == "person-hours"` (÷ 160)
- [x] 5.2 Update the confidence range label to display "± {value:.1f} person-months" if CV MAE is available; hide/omit the label if `cv_mae` is absent
- [x] 5.3 On `_on_reset`, revert the confidence range label to "—" (or blank)

## 6. Right Panel — Remove Chart

- [x] 6.1 Remove the matplotlib chart panel and related `_draw_chart`, `_draw_empty_chart`, `_embed_chart` methods from `app.py`
- [x] 6.2 Remove matplotlib imports if no longer used elsewhere; remove `FigureCanvasTkAgg` import
- [x] 6.3 Adjust window geometry and column weights to fill the freed right-column space with the trust dashboard or collapse to a two-column layout

## 7. Cost / Team Calculator — Wiring Update

- [x] 7.1 Ensure `_update_calculator` reads the estimate from `_on_estimate`'s result (single value) rather than `self._all_preds.get(selected)` — update the internal state variable accordingly
- [x] 7.2 Verify Rp cost and team size outputs still update correctly after estimation and on gaji/durasi input change

## 8. Smoke Testing

- [ ] 8.1 Launch the app: verify no dataset/model dropdowns are visible, input form renders the correct fields for the auto-selected dataset
- [ ] 8.2 Enter valid feature values and press Estimasi: verify effort estimate, quality label, confidence range, and metric table all populate
- [ ] 8.3 Expand Detail Teknis: verify dataset name, model name, and metrics are shown correctly
- [ ] 8.4 Press Reset Form: verify estimate, confidence range, and metric table revert to "—"
- [ ] 8.5 Enter gaji and durasi after estimation: verify Rp total and team size update correctly
- [ ] 8.6 Delete all `*_metrics.json` files temporarily and restart: verify the error banner is shown and Estimasi is disabled
