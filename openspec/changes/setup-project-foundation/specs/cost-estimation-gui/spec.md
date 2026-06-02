## ADDED Requirements

### Requirement: Load serialized models on startup
The GUI SHALL load `models/svr_model.pkl`, `models/lr_model.pkl`, and `models/scaler.pkl` at application startup using `joblib.load`. If any file is missing, the GUI SHALL display a clear error message dialog and disable the prediction controls rather than crash.

#### Scenario: All model files present
- **WHEN** the application starts and all three `.pkl` files exist
- **THEN** the GUI launches normally with prediction controls enabled

#### Scenario: Model files missing
- **WHEN** one or more `.pkl` files are not found at startup
- **THEN** the GUI displays an error message instructing the user to run `model_pipeline.py` first, and the "Estimasi" button is disabled

---

### Requirement: Project metrics input form
The GUI SHALL provide labeled input fields for the following project metrics: KLOC (estimated lines of code in thousands), reliability (effort multiplier), complexity (effort multiplier), team size, and required development schedule. All fields SHALL accept numeric input only.

#### Scenario: Valid numeric input accepted
- **WHEN** the user enters a numeric value in any input field
- **THEN** the value is accepted and retained until changed or reset

#### Scenario: Non-numeric input rejected
- **WHEN** the user enters a non-numeric value (e.g., letters, symbols) in an input field and presses "Estimasi"
- **THEN** the GUI displays an inline error message identifying the invalid field and does NOT call the model

---

### Requirement: Model selection
The GUI SHALL provide a selector (Combobox or SegmentedButton) allowing the user to choose between SVR and Linear Regression for the prediction.

#### Scenario: SVR selected
- **WHEN** the user selects "SVR" and clicks "Estimasi"
- **THEN** the prediction is computed using the loaded SVR model

#### Scenario: Linear Regression selected
- **WHEN** the user selects "Linear Regression" and clicks "Estimasi"
- **THEN** the prediction is computed using the loaded LinearRegression model

---

### Requirement: Instant prediction output
After the user clicks "Estimasi", the GUI SHALL scale inputs using the loaded scaler, run inference with the selected model, and display the predicted effort (Person-Months) prominently in the output area within the same window without opening a new dialog.

#### Scenario: Successful prediction
- **WHEN** all inputs are valid and the user clicks "Estimasi"
- **THEN** the predicted effort value appears in the output label immediately

---

### Requirement: Reset form
The GUI SHALL provide a "Reset Form" button that clears all input fields and the output area, returning the form to its initial empty state, without restarting the application.

#### Scenario: Reset clears all fields and output
- **WHEN** the user clicks "Reset Form"
- **THEN** all input fields are cleared to empty and the output label is cleared

---

### Requirement: Dark theme UI
The GUI SHALL use CustomTkinter's dark appearance mode with a muted color palette (dark gray background, `gray17` or equivalent). The layout SHALL use `CTkFrame` containers with a `grid` geometry manager for structured alignment.

#### Scenario: Dark mode enforced at launch
- **WHEN** the application starts
- **THEN** `customtkinter.set_appearance_mode("dark")` is called before the main window is shown, and no light-mode widgets are visible
