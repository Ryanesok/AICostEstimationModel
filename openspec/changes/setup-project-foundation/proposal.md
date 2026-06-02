## Why

OpenSpec needs a working foundation before any project manager can use it. Currently the repository is empty — no ML pipeline, no GUI, no project structure. This change establishes the complete runnable base of the application so the cost-estimation tool can be demonstrated and iterated on.

## What Changes

- Add `requirements.txt` with all Python dependencies (scikit-learn, pandas, numpy, customtkinter, joblib)
- Add `model_pipeline.py`: modular ML training pipeline that loads/generates COCOMO-structured data, preprocesses it, trains SVR and Linear Regression models, evaluates with MAE and R², and exports `.pkl` artifacts
- Add `app.py`: offline desktop GUI using customtkinter that loads trained `.pkl` models and provides an interactive project-metrics form with instant cost/effort prediction output
- Add project folder structure (`models/`, `data/`) to organize outputs cleanly

## Capabilities

### New Capabilities

- `model-training-pipeline`: Loads COCOMO-structured CSV data (or falls back to built-in dummy data), preprocesses with StandardScaler, trains SVR and LinearRegression models, evaluates with MAE and R², and serializes model + scaler to `models/*.pkl`
- `cost-estimation-gui`: CustomTkinter dark-theme desktop app that loads serialized models, accepts project metrics as input (KLOC, effort multipliers, etc.), displays instant cost/effort predictions, and supports form reset without restarting

### Modified Capabilities

<!-- No existing specs to modify — this is a greenfield setup -->

## Impact

- **New files:** `requirements.txt`, `model_pipeline.py`, `app.py`, `models/` dir, `data/` dir
- **Dependencies:** scikit-learn, pandas, numpy, customtkinter, joblib
- **Runtime:** 100% offline — no network calls, no cloud services
- **Interoperability:** `model_pipeline.py` must be run once before `app.py` to generate `.pkl` files; `app.py` handles missing model files gracefully with an error message
