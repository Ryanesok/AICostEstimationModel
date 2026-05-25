## Why

The app currently exposes its internals — dataset selection, model picker, raw per-model predictions — to project managers who have no ML background and no need to choose between SVR, XGBoost, or the COCOMO-81 dataset. This creates decision fatigue and erodes trust: the manager cannot know which choice gives the most accurate result. The opportunity is to flip the UI around the manager's actual job: get a single, validated effort estimate they can confidently defend.

## What Changes

- Remove the dataset and model selection dropdowns from the user-facing UI; the system selects the best-performing model/dataset combination automatically.
- Introduce an **auto-selection engine** that picks the dataset and model with the best cross-validated MAE and MMRE from the stored metrics files.
- Replace the multi-model comparison chart with a **trust dashboard**: shows the selected model's evaluation metrics (MAE, MMRE, PRED(25)) with plain-language quality labels ("Akurasi tinggi", "Cukup baik", etc.).
- Add a **confidence indicator** next to the result that communicates how far the estimate might deviate (e.g., ±N person-months based on CV MAE).
- Keep the cost/team calculator unchanged; it now feeds from the auto-selected estimate.
- Retain a collapsible "Detail Teknis" section for power users who still want to inspect which dataset and model were chosen and why.

## Capabilities

### New Capabilities

- `auto-model-selection`: Logic that reads all `*_metrics.json` files, scores each dataset/model combination, and returns the best configuration for a given set of input features — without user intervention.
- `estimation-trust-dashboard`: UI panel that presents evaluation metrics (MAE, MMRE, PRED(25)) in a project-manager-friendly format alongside the estimate result, so users understand result reliability without ML knowledge.
- `confidence-interval-display`: Displays a ±range derived from the model's CV MAE next to the effort estimate, giving the manager a realistic precision expectation.

### Modified Capabilities

<!-- No existing openspec specs exist yet; all capabilities are new. -->

## Impact

- **app.py**: UI restructured — dataset/model dropdowns removed from the default view; auto-selection drives the estimate flow; middle panel rebuilt around the trust dashboard.
- **model_pipeline.py**: No changes required; `*_metrics.json` files already contain all needed evaluation data.
- **dataset_config.yaml / field_labels.yaml**: Input form must still render dataset-specific fields; auto-selection must map available metrics to the correct field config.
- **New module** (`estimator.py` or similar): Encapsulates auto-selection logic, separating it cleanly from UI code.
- No breaking changes to the training pipeline or stored model artifacts.
