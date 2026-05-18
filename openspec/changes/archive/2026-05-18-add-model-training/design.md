## Context

The current codebase (`setup-project-foundation`) has a single `model_pipeline.py` that trains on one fixed dataset and exports two `.pkl` files with hardcoded names (`svr_model.pkl`, `lr_model.pkl`). The `app.py` has a hardcoded 5-field form and a toggle between those two models.

The v2 spec introduces a three-script separation: a downloader, a batch-training pipeline, and a new GUI. The key architectural challenge is handling datasets with different column names (COCOMO81 uses `Actual_Effort`, Desharnais uses `Effort`, etc.) without requiring code changes per dataset.

## Goals / Non-Goals

**Goals:**
- Introduce a URL registry (`datasets.txt`) + `downloader.py` that fetches CSVs into `data/` non-destructively
- Rewrite `model_pipeline.py` to iterate every CSV in `data/`, apply a per-dataset column-mapping config, and export named `.pkl` pairs (`svr_<dataset>.pkl`, `linreg_<dataset>.pkl`)
- Rewrite `app.py` to dynamically discover all available `.pkl` pairs in `models/` and expose them via a dropdown; add inline cost/team calculator below the prediction result
- Preserve the 100% offline runtime guarantee (downloader is setup-only)

**Non-Goals:**
- Auto-detecting column structure from unknown CSVs without config (out of scope; config-driven mapping only)
- Retraining from within the GUI
- Cloud sync or multi-user deployment

## Decisions

### 1. Column-mapping dictionary in `model_pipeline.py`

**Decision:** Embed a `DATASET_CONFIG` dict keyed by filename stem (e.g. `"cocomo81"`) containing `features` (list of column names) and `target` (column name). The pipeline looks up config by matching the CSV filename stem.

**Rationale:** Simple, no external config files to maintain; keeps mapping close to training logic; easy to extend by adding a new dict entry.

**Alternative considered:** Auto-detect target by heuristic (e.g. column named "effort" case-insensitive). Rejected — fragile across dataset variants and harder to audit.

---

### 2. Artifact naming: `svr_<stem>.pkl` / `linreg_<stem>.pkl`

**Decision:** Use the CSV filename stem (lowercased, stripped of non-alphanumeric chars) as the model identifier. Exported as `models/svr_cocomo81.pkl` and `models/linreg_cocomo81.pkl`.

**Rationale:** Pairs are trivially discoverable by glob pattern `models/svr_*.pkl`; the GUI can list available datasets by stripping the prefix.

**Alternative considered:** Single directory-per-dataset (`models/cocomo81/svr.pkl`). Rejected — more filesystem complexity for minimal benefit.

---

### 3. GUI dynamic model discovery

**Decision:** At startup, `app.py` scans `models/svr_*.pkl` to build the list of available datasets. The CTkOptionMenu is populated from this scan. Selecting a dataset loads `svr_<name>.pkl`, `linreg_<name>.pkl`, and `scaler_<name>.pkl`.

**Rationale:** No hardcoding — if the user adds a new dataset and retrains, the GUI automatically shows it on next launch.

---

### 4. `requests` for downloader (not `urllib`)

**Decision:** Use `requests` library for `downloader.py`.

**Rationale:** `requests` provides cleaner error handling, streaming downloads, and timeout control compared to `urllib`. The project spec lists both; `requests` is the better-maintained option for file downloads.

---

### 5. Inline cost/team calculator — same window, collapsible section

**Decision:** Add a calculator section below the prediction result in the left panel. It appears after a successful estimate with two additional inputs (monthly salary in Rupiah, target duration in months) and two computed outputs (total cost, team size).

**Rationale:** Keeps the workflow in one window; the project spec calls for "kalkulator kecil di dalam UI". Collapsing it until a prediction exists avoids cluttering the initial view.

## Risks / Trade-offs

- **Unknown CSV format** → Pipeline logs a warning and skips the file. Mitigation: `DATASET_CONFIG` must be updated for any new dataset added to `datasets.txt`.
- **Scaler naming change** — v1 used `scaler.pkl`; v2 uses `scaler_<name>.pkl`. Existing `.pkl` files from v1 are incompatible. Mitigation: documented in migration plan; re-run pipeline after upgrade.
- **`requests` adds a network-phase dependency** → only at download time, not at GUI runtime. Mitigation: downloader wraps every request in try/except; offline use is unaffected.

## Migration Plan

1. Run `pip install -r requirements.txt` (adds `requests`)
2. Populate `datasets.txt` with desired CSV URLs (or use provided defaults)
3. Run `python downloader.py` once to fetch CSVs into `data/`
4. Run `python model_pipeline.py` to regenerate all `.pkl` files under new naming scheme
5. Delete old `models/svr_model.pkl`, `models/lr_model.pkl`, `models/scaler.pkl` (now superseded)
6. Launch `python app.py` — dropdown will show all newly trained datasets
