## Context

OpenSpec is a greenfield Python desktop application for offline software cost estimation. The repository currently contains only a project overview document. Two developers (project manager end-users + the builder) will interact with the system: one runs `model_pipeline.py` once to train and serialize models, another (or the same person) runs `app.py` daily to make predictions.

The system must never call the network — all ML inference happens locally from `.pkl` files loaded at startup.

## Goals / Non-Goals

**Goals:**
- Establish runnable project scaffold (folder layout, `requirements.txt`)
- Implement a modular, testable `model_pipeline.py` that produces `.pkl` artifacts from COCOMO-structured data
- Implement a `customtkinter` dark-theme desktop GUI (`app.py`) that loads those artifacts and delivers instant predictions
- Ensure graceful error handling: missing model files, non-numeric input, empty fields

**Non-Goals:**
- Real-time model retraining from within the GUI
- Multi-user / networked deployment
- Support for non-COCOMO dataset formats in this iteration
- Hyperparameter tuning UI

## Decisions

### 1. Two-script separation (pipeline vs. GUI)

**Decision:** Keep `model_pipeline.py` and `app.py` as separate, independently runnable scripts.

**Rationale:** Separation of concerns — training is a one-time (or periodic) offline operation; inference happens frequently. Mixing them forces unnecessary re-training on every app launch and bloats GUI startup time.

**Alternative considered:** Single monolithic script. Rejected because it couples training latency to every user session.

---

### 2. Both SVR and LinearRegression in one pipeline

**Decision:** Train both models in a single pipeline run and export both `.pkl` files. The GUI lets the user select which model to use.

**Rationale:** Lets the user compare predictions from a non-parametric (SVR) and a parametric (LinearRegression) model without needing to re-run the pipeline. MAE and R² are printed at training time so the user can see which model performs better on training data.

**Alternative considered:** Train only the best model automatically. Rejected because the project spec explicitly lists both models and hiding the choice reduces transparency.

---

### 3. Dummy data fallback in `model_pipeline.py`

**Decision:** If no CSV is found at the expected path, `model_pipeline.py` generates synthetic COCOMO-structured data inline and proceeds normally.

**Rationale:** Enables end-to-end testing and demonstration without requiring the user to download the PROMISE dataset first.

**Alternative considered:** Hard-fail if CSV is missing. Rejected because it blocks first-run experience and testing.

---

### 4. `joblib` for serialization (not `pickle`)

**Decision:** Use `joblib.dump` / `joblib.load` for all `.pkl` files.

**Rationale:** `joblib` is the recommended serializer for scikit-learn objects — more efficient for large numpy arrays embedded in models. The project overview lists both, but joblib is strictly superior for this use-case.

---

### 5. `customtkinter` with `gray17` dark theme, grid layout

**Decision:** Use `CTk` appearance mode `"dark"`, color theme `"blue"`, and `grid` geometry manager throughout.

**Rationale:** The project spec explicitly mandates dark theme. Grid provides predictable column alignment for form fields vs. labels. Pack is simpler but harder to align multi-column forms.

## Risks / Trade-offs

- **`.pkl` version drift** → If scikit-learn is upgraded, serialized models may be incompatible. Mitigation: pin scikit-learn version in `requirements.txt`; re-run pipeline after any upgrade.
- **Synthetic dummy data** → Model trained on dummy data produces meaningless predictions. Mitigation: clearly log "DUMMY DATA" warning at training time; instruct user to replace with real CSV.
- **Single-threaded GUI** → Long prediction calls block the UI. Mitigation: SVR inference on tabular data is sub-millisecond at this scale, so threading is unnecessary for now.
- **No input sanitization beyond type checking** → Extremely large KLOC values may produce out-of-distribution predictions. Mitigation: add reasonable upper-bound validation in the GUI (future iteration).
