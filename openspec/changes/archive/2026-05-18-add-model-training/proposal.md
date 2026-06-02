## Why

The current system is trained on a single hardcoded dataset with fixed column names. The v2 spec calls for a multi-dataset architecture where any number of PROMISE-repository CSVs (COCOMO81, Desharnais, etc.) can be downloaded automatically and trained independently — giving project managers the ability to choose the best-fit model for their project context.

## What Changes

- Add `datasets.txt` — plain-text registry of raw CSV URLs to download
- Add `downloader.py` — reads `datasets.txt`, downloads each file into `data/`, skips if already present, handles network errors gracefully without crashing
- **BREAKING** Replace `model_pipeline.py` — rewrite from single-dataset to multi-dataset batch training with a column-mapping dictionary per dataset; exports one SVR and one LR model per dataset (e.g. `svr_cocomo81.pkl`, `linreg_desharnais.pkl`)
- **BREAKING** Replace `app.py` — replace fixed input form with a dynamic model-selector dropdown (`CTkOptionMenu`); add inline cost calculator that converts Person-Months → total cost in Rupiah and required team size

## Capabilities

### New Capabilities

- `dataset-downloader`: Reads a URL registry (`datasets.txt`), downloads CSV files into `data/`, non-destructive (skip existing), fault-tolerant on network errors
- `multi-dataset-pipeline`: Batch-trains SVR + LinearRegression on every CSV in `data/` using a per-dataset column-mapping config; exports named `.pkl` pairs per dataset into `models/`
- `dynamic-model-selector-gui`: Dynamic CTkOptionMenu that lists all available trained models; form adapts to selected dataset's feature set; adds inline calculator for cost (Rupiah) and team-size conversion

### Modified Capabilities

- `model-training-pipeline`: Requirements change — single fixed dataset → batch multi-dataset with column-mapping dictionary and per-dataset artifact naming
- `cost-estimation-gui`: Requirements change — static form + dual-model toggle → dynamic dataset selector + inline cost/team calculator panel

## Impact

- **Replaced files:** `model_pipeline.py`, `app.py`
- **New files:** `downloader.py`, `datasets.txt`
- **New dependencies:** `requests` (for downloader)
- **Breaking:** existing `models/*.pkl` naming convention changes; users must re-run the pipeline after this upgrade
- **Offline guarantee preserved:** downloader is a one-time setup step; `app.py` still loads only local `.pkl` files at runtime
