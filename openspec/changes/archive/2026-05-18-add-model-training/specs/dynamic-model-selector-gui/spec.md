## ADDED Requirements

### Requirement: Dynamic dataset discovery at startup
The GUI SHALL scan `models/svr_*.pkl` at startup to build the list of available trained datasets. If no models are found, it SHALL display an error state.

#### Scenario: Models present — dropdown populated
- **WHEN** one or more `models/svr_*.pkl` files exist at startup
- **THEN** the CTkOptionMenu lists each dataset stem (e.g. "cocomo81", "desharnais") and selects the first automatically

#### Scenario: No models found
- **WHEN** `models/` contains no `svr_*.pkl` files
- **THEN** the GUI displays an error banner instructing the user to run `downloader.py` then `model_pipeline.py`, and disables the "Estimasi" button

---

### Requirement: Load model trio on dataset selection
When the user selects a dataset from the dropdown, the GUI SHALL load the corresponding `svr_<name>.pkl`, `linreg_<name>.pkl`, and `scaler_<name>.pkl` from `models/`.

#### Scenario: Dataset switched mid-session
- **WHEN** the user changes the dropdown selection
- **THEN** the three model files for the new selection are loaded and the form is reset

---

### Requirement: Inline cost and team-size calculator
Below the Person-Months prediction result, the GUI SHALL provide two additional input fields — average monthly salary (Rupiah) and target duration (months) — and display the computed total cost and required team size.

#### Scenario: Calculator updates after estimate
- **WHEN** a prediction is shown and the user fills in salary and duration
- **THEN** Total Cost = prediction × salary and Team Size = ceil(prediction / duration) are computed and displayed

#### Scenario: Calculator cleared on reset
- **WHEN** the user clicks "Reset Form"
- **THEN** salary and duration inputs are cleared and calculator outputs return to "—"

---

### Requirement: Model toggle within selected dataset
The GUI SHALL provide a toggle (CTkSegmentedButton) to switch between SVR and Linear Regression predictions for the currently selected dataset, both using the same loaded scaler.

#### Scenario: Toggle switches active prediction
- **WHEN** the user switches between SVR and Linear Regression
- **THEN** the displayed Person-Months value updates immediately without re-running the full estimate flow
