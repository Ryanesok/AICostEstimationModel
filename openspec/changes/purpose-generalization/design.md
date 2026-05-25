## Context

The current `app.py` is tightly coupled to the choice of dataset and model: the user must select both before getting a result. Evaluation metrics exist per dataset/model in `models/*_metrics.json`, but they are only used internally during training. The UI exposes all models simultaneously (comparison chart) rather than surfacing a single authoritative answer.

The target user — a project manager — needs one reliable estimate with an honest quality statement. They have no training to interpret why SVR gives 8.3 person-months while XGBoost gives 7.1. Forcing that choice on them is a UX failure and a trust failure.

Constraints:
- All model artifacts and metrics files are pre-trained and stored on disk. No retraining at runtime.
- The input form must remain dataset-specific (each dataset has different feature columns).
- The application is a local desktop app (CustomTkinter); no network calls.

## Goals / Non-Goals

**Goals:**
- Auto-select the best dataset/model pair using stored CV metrics (MAE as primary criterion, MMRE as tiebreaker).
- Present the estimate with a plain-language quality indicator and a ±confidence range.
- Show evaluation metrics (MAE, MMRE, PRED(25)) in the result panel so managers can verify reliability.
- Provide a collapsible technical detail section for transparency without cluttering the main flow.
- Keep the cost/team calculator working from the auto-selected estimate.

**Non-Goals:**
- Retraining or re-evaluating models at runtime.
- Supporting user-defined custom datasets or models at runtime.
- Multi-project or batch estimation.
- Changing the training pipeline (`model_pipeline.py`, `downloader.py`, `auto_configure.py`).

## Decisions

### D1: Auto-selection scoring — MAE-first, MMRE as tiebreaker

Each `*_metrics.json` contains per-model CV MAE. The auto-selector iterates all metrics files and all model keys within each, builds a scored list, and picks the minimum CV MAE. MMRE (Mean Magnitude of Relative Error) breaks ties if two models are within 1% MAE of each other.

**Why not rank by R²?** R² is scale-dependent and can be misleading on small datasets (some datasets have R² < 0.4 even for practically useful models). MAE is directly interpretable by a project manager ("typically off by N person-months").

**Why not user-configurable weighting?** The proposal explicitly removes model/dataset decisions from the user. A hidden heuristic is better than a hidden heuristic that the user must configure.

### D2: New `estimator.py` module for selection logic

Auto-selection and metric lookup are extracted into a standalone `estimator.py` (or `core/estimator.py`) module. `app.py` imports it and calls a single function: `select_best_estimator(metrics_dir) -> EstimatorResult`.

**Why a separate module?** Keeps `app.py` focused on UI. Makes the selection logic unit-testable without launching the GUI. Avoids a god-object anti-pattern.

### D3: Input form still driven by the selected dataset's field config

Since different datasets have different features, the form must be rebuilt when the auto-selected dataset changes. On first load, `estimator.py` returns the best dataset stem, and `app.py` loads that dataset's field config. The user never needs to know which dataset was picked — it is shown only in the collapsed technical detail section.

**Alternative considered:** Show all features from all datasets. Rejected — field sets are incompatible across datasets (e.g., function points vs. COCOMO cost drivers).

### D4: Trust dashboard replaces the multi-model comparison chart

The right panel is replaced by a trust dashboard showing:
1. **Effort estimate** (large, prominent) with unit
2. **Confidence range**: estimate ± CV MAE (e.g., "±3.2 person-months")
3. **Quality label**: derived from MMRE thresholds (MMRE ≤ 0.25 → "Akurasi tinggi", ≤ 0.50 → "Cukup baik", > 0.50 → "Perlu verifikasi manual")
4. **Metric table**: MAE, MMRE, PRED(25) with brief explanations
5. **Collapsible "Detail Teknis"**: dataset name, model name, best hyperparameters, raw CV scores

**Why remove the model comparison chart?** It answered "which model performs best?" — a developer question, not a manager question. The trust dashboard answers "can I trust this number?" — the manager question.

### D5: MMRE and PRED(25) computed from stored metrics

`*_metrics.json` already stores `cv_mae_mean` per model. MMRE is not currently stored. Two options:
- **Option A**: Add MMRE computation to `model_pipeline.py` and store it.
- **Option B**: Derive an approximate MMRE from MAE and the dataset's mean effort (loadable from the CSV at runtime).

Decision: **Option A** is cleaner — compute and store MMRE and PRED(25) during training. This requires a one-time re-run of `model_pipeline.py` after the metric keys are added. PRED(25) = fraction of predictions where relative error ≤ 0.25.

## Risks / Trade-offs

- [Risk] Auto-selection always picks the same dataset regardless of the project's actual context → Mitigation: The technical detail section shows which dataset was chosen; a future iteration can add a "domain hint" toggle (not in scope here).
- [Risk] `model_pipeline.py` re-run required to populate new metric keys (MMRE, PRED(25)) → Mitigation: The auto-selector gracefully falls back to MAE-only scoring when these keys are absent; the trust dashboard shows "N/A" for unavailable metrics.
- [Risk] If all metrics files are missing or corrupted, auto-selection has no basis → Mitigation: Fall back to the existing error banner ("Model tidak tersedia") with instructions to re-run the pipeline.
- [Trade-off] Removing user model selection reduces flexibility for ML-savvy users → Accepted; the collapsed technical detail section preserves transparency without forcing the choice on everyone.

## Migration Plan

1. Add MMRE and PRED(25) computation to `model_pipeline.py` (new metrics keys in JSON output).
2. Implement `estimator.py` with `select_best_estimator()` and `load_estimator()`.
3. Refactor `app.py`: remove dataset/model dropdowns, wire auto-selection on startup, rebuild middle and right panels.
4. Re-run `model_pipeline.py` to regenerate metrics files with new keys.
5. Smoke-test locally: launch app, verify form loads, verify estimate renders, verify trust dashboard populates.

Rollback: the old `app.py` is preserved in git history. No database or file format changes affect the training pipeline.

## Open Questions

- Should the "Detail Teknis" section be expanded by default for the first run, then collapsed on subsequent runs? (User preference persistence not yet in scope.)
- PRED(25) threshold — is 25% the right tolerance for this domain, or should it be PRED(30)? Literature uses both; default to 25% for now and make it a named constant.
