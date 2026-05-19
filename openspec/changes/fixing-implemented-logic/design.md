## Context

`model_pipeline.py` currently trains SVR and LinearRegression with a fixed 80/20 `train_test_split(random_state=42)`. On the smallest dataset (COCOMO-81, ~63 usable rows) the test set is ~12 rows — so a single MAE/R² reading carries high variance. SVR is trained with default `SVR(kernel="rbf")` and no hyperparameter search; a poor choice of `C` or `epsilon` can dominate error. `auto_configure.py` accepts any numeric column as a feature with no check for correlation to the target — making accidental data leakage invisible. Evaluation results are only printed; nothing is persisted.

## Goals / Non-Goals

**Goals:**
- Produce reliable performance estimates via k-fold cross-validation instead of a single split
- Tune SVR hyperparameters per dataset using `GridSearchCV`
- Detect and surface potential data-leakage columns in `auto_configure.py` before the config is saved
- Persist evaluation metrics to `models/<stem>_metrics.json` for auditability
- Train the final production model on the full dataset (after CV tuning confirms the approach)

**Non-Goals:**
- Changing `LinearRegression` (OLS has no hyperparameters to tune)
- Adding non-linear feature engineering or dimensionality reduction
- Changing `app.py` — model `.predict()` API is unchanged
- Replacing `StandardScaler` (scaling strategy is correct)
- Adding automated feature selection beyond the leakage guard

## Decisions

**Decision: k-fold CV on the full dataset; production model trained on full data**

Alternatives:
- *Keep train/test split, add repeated splits*: still high variance at n=63
- *Nested CV (outer loop for eval, inner for tuning)*: statistically ideal but slow and complex for this dataset size
- *Chosen — KFold(k=5) for evaluation + GridSearchCV for tuning; final pkl trained on 100% of data*: gives stable metrics, maximises training data for the deployed model. Trade-off: final model is not evaluated on truly held-out data, but CV score is the honest estimate.

**Decision: GridSearchCV param grid scoped to avoid exponential blowup**

Grid: `C ∈ [0.1, 1, 10, 100]`, `epsilon ∈ [0.01, 0.1, 1.0]`, `kernel ∈ ['rbf', 'linear']` → 24 combinations × 5 folds = 120 fits per dataset. Larger grids (e.g., adding `poly`) add minutes of training time for marginal gain on these small datasets.

**Decision: Leakage guard uses Pearson |r| > 0.95 threshold**

A correlation above 0.95 with the target almost certainly means the feature is derived from or identical to the target (e.g., `Effort` as a feature when predicting `Duration` in the same table). Threshold is configurable as a constant; lower values produce too many false-positive warnings on legitimately correlated engineering metrics.

**Decision: `auto_configure.py` warns but does not block**

The tool is interactive; it informs the user of suspicious columns and asks for confirmation to keep them. Blocking outright would be wrong for legitimate domain knowledge the tool cannot model.

**Decision: Metrics saved to `models/<stem>_metrics.json`**

Stored alongside pkl files so they can be read by the app or CI scripts later without re-training. Content: `cv_mae_mean`, `cv_mae_std`, `cv_r2_mean`, `cv_r2_std`, `best_svr_params`, `lr_cv_mae_mean`, `lr_cv_r2_mean`, `trained_on_rows`.

## Risks / Trade-offs

- [Risk] GridSearchCV makes training ~10-20x slower per dataset → Mitigation: grid is small (24 params × 5 folds); all 4 datasets together should finish in < 60 seconds on a modern CPU
- [Risk] Training on full data after CV means there is no separate test set in the pkl workflow → Mitigation: CV score is the documented estimate; metrics JSON explicitly labels values as `cv_*` to avoid confusion
- [Risk] Leakage guard misidentifies legitimate domain features (e.g., AFP and Duration may genuinely correlate) → Mitigation: guard only warns, requires user confirmation; threshold at 0.95 is conservative

## Migration Plan

1. Update `model_pipeline.py`: `preprocess()` removes the train/test split (returns full X, y + fitted scaler); new `cross_validate_models()` handles CV + GridSearch; `train_and_export()` calls CV for metrics, then fits final model on full data
2. Update `auto_configure.py`: `configure_one()` adds a leakage-check step after target selection, before saving
3. Re-run `python model_pipeline.py` to regenerate all pkl files and produce metrics JSON
4. No rollback needed — pkl file names are unchanged; app.py is untouched
