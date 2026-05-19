## Context

`model_pipeline.py` currently trains SVR (via GridSearchCV) and LinearRegression, saves two pkl files per dataset, and writes a `_metrics.json`. `app.py` loads the two pkl files at startup, selects the active model via a `CTkSegmentedButton` with two options, and draws a 2-bar horizontal chart. All model loading is keyed by `self._stems[i]` with hardcoded file prefixes `svr_` and `linreg_`.

New models: Random Forest (sklearn), XGBoost (xgboost package), LSTM (PyTorch single-timestep), and Hybrid (weighted average at inference time).

## Goals / Non-Goals

**Goals:**
- Train RF, XGBoost, LSTM per dataset in `model_pipeline.py`; export pkl + update metrics JSON
- Compute Hybrid prediction at inference in `app.py` using pre-saved model weights from metrics JSON
- Replace model toggle `CTkSegmentedButton` with `CTkOptionMenu` supporting all 6 model names
- Chart shows all 6 model bars; selected model bar highlighted, others muted
- Add `xgboost` and `torch` to `requirements.txt`

**Non-Goals:**
- Hyperparameter tuning for RF or LSTM (RF uses good defaults; LSTM uses a fixed architecture)
- Time-series / sequence exploitation in LSTM (data is tabular; single-timestep is the design choice)
- Stacking meta-learner for Hybrid (weighted average is sufficient and interpretable)
- Persisting LSTM as a `.pkl` via joblib (PyTorch models use `.pt` state dict instead)

## Decisions

**Decision: LSTM architecture — single hidden layer, 32 units, single-step input**

Input shape: `(batch, seq=1, features)`. One LSTM layer (hidden=32), one linear output layer. Trained with Adam, MSE loss, 200 epochs, batch_size=32. No data augmentation.

Alternatives considered: multi-layer LSTM (overkill for ≤499 rows), MLP instead of LSTM (user explicitly requested LSTM), sklearn `MLPRegressor` (no LSTM cells). Trade-off: single-step LSTM is mathematically equivalent to a dense layer over LSTM hidden state — it adds complexity for marginal gain on tabular data, but satisfies the intent.

**Decision: LSTM saved as `lstm_<stem>.pt` + `lstm_arch_<stem>.json` (not pkl)**

PyTorch `state_dict` serializes cleanly with `torch.save`/`torch.load`. Joblib pkl works for sklearn objects; PyTorch models should use native serialization. `lstm_arch_<stem>.json` stores `input_size` needed to reconstruct the model at load time.

**Decision: Hybrid weights computed from CV MAE, stored in `_metrics.json`**

At inference, `app.py` reads `hybrid_weights` from `<stem>_metrics.json`: `{ "lstm": w1, "xgb": w2, "lr": w3 }` where `w_i = (1/mae_i) / sum(1/mae_j)`. No separate pkl needed — Hybrid is a pure inference-time computation.

**Decision: Model selector changes from `CTkSegmentedButton` to `CTkOptionMenu`**

`CTkSegmentedButton` with 6 entries becomes unreadably narrow in a 380px middle panel. `CTkOptionMenu` with the same `self._model_var` StringVar drops in as a direct replacement with no logic changes needed downstream.

**Decision: Chart highlight — selected bar uses theme accent color, others use `#555555`**

Current chart draws all bars in one color. New: iterate all 6 bars; if `bar_label == selected_model`, use `#4a9eff` (accent); otherwise `#555555` (muted). No other chart logic changes.

**Decision: RF uses `n_estimators=200, random_state=42`, no GridSearchCV**

RF is robust to hyperparameter choices and already averages over trees internally. GridSearchCV on RF adds 3–5 minutes training time per dataset for marginal gain; not worth it for this use case.

**Decision: XGBoost uses a small fixed grid: `n_estimators ∈ [100, 300]`, `max_depth ∈ [3, 5]`, `learning_rate ∈ [0.05, 0.1]`**

12 combinations × 5 folds = 60 fits. Balances tuning quality with training time (~10–30s per dataset).

## Risks / Trade-offs

- [Risk] `torch` is a large dependency (~2 GB) → Mitigation: note in requirements that CPU-only torch (`torch --index-url https://download.pytorch.org/whl/cpu`) is sufficient
- [Risk] LSTM with 63 rows (COCOMO-81) will overfit → Mitigation: early stopping on a 20% validation split during LSTM training; accept that LSTM metrics on small datasets will be honest but poor
- [Risk] Hybrid weight computation fails if any component model has MAE=0 → Mitigation: add `max(mae, 1e-6)` floor before inversion
- [Risk] Chart with 6 bars becomes visually crowded → Mitigation: use horizontal bars (already horizontal), reduce font size, truncate long model names to abbreviations

## Migration Plan

1. Add `xgboost` and `torch` to `requirements.txt`
2. Add `train_rf()`, `train_xgb()`, `train_lstm()` functions to `model_pipeline.py`
3. Update `train_and_export()` to call all five model trainers; extend metrics JSON with new metrics + hybrid weights
4. Update `cleanup_stale_models()` to also remove `xgb_`, `rf_`, `lstm_`, `lstm_arch_` files
5. Update `app.py`: replace model toggle widget; update `_load_dataset()` to load new pkl/pt files; update `_on_estimate()` to dispatch; update chart to render 6 bars with highlight
6. Re-run `python model_pipeline.py` to generate new pkl/pt files
