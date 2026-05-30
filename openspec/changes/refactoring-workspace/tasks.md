## 1. Package Scaffolding

- [x] 1.1 Create `core/__init__.py` (empty, marks package)
- [x] 1.2 Create `ui/__init__.py` (empty, marks package)
- [x] 1.3 Create `explainer/__init__.py` (empty, marks package)
- [x] 1.4 Verify all three packages importable from project root (`import core`, `import ui`, `import explainer`)

## 2. PM Input Schema (`core/schema.py`)

- [x] 2.1 Define `PMInput` dataclass with all 15 fields across four dimensions (Technical, Team, Business, Risk)
- [x] 2.2 Add field validation: `feature_complexity`, `seniority`, `budget_constraint`, `requirement_change_risk`, `uncertainty_level`, `technical_debt`, `security_level`, `stack_experience` must be one of their allowed string values
- [x] 2.3 Add range validation: `num_developers >= 1`, `deadline_months > 0`, `external_dependencies >= 0`, `num_features >= 1`, `num_apis >= 0`
- [x] 2.4 Expose `PMInput` from `core/__init__.py`

## 3. Estimator Bridge (`core/estimator_bridge.py`)

- [x] 3.1 Implement `estimate(pm_input: PMInput) -> BridgeResult` — maps PM fields to ML feature vector and calls `pipeline.estimator.run_estimate()`
- [x] 3.2 Document each field mapping in inline comments (e.g., `feature_complexity` -> synthetic `adjusted_fp` score)
- [x] 3.3 Implement `simulate(base_input: PMInput, field: str, value) -> BridgeResult` — creates a copy with one field overridden and calls `estimate`
- [x] 3.4 Implement `compare(base, simulated) -> dict` — returns `delta_effort`, `delta_pct`, `direction`

## 4. Confidence Module (`core/confidence.py`)

- [x] 4.1 Implement `compute(result: BridgeResult) -> dict` returning `lower`, `upper`, `confidence_pct`, `range_pct`
- [x] 4.2 Base confidence on model's pred25 and mmre metrics from BridgeResult; document formula in comments
- [x] 4.3 Clamp `confidence_pct` to [0, 100]

## 5. Dominance Module (`core/dominance.py`)

- [x] 5.1 Implement `compute(pm_input: PMInput, result: BridgeResult) -> dict` returning effort share per dimension
- [x] 5.2 Ensure returned shares sum to 100 (±0.1)
- [x] 5.3 Define scoring weights per dimension (Technical, Team, Business, Risk) based on field values; document weighting logic

## 6. Result Explainer (`explainer/result_explainer.py`)

- [x] 6.1 Implement `explain(pm_input, estimator_result, dominance) -> str` returning a human-readable narrative
- [x] 6.2 Include dominant dimension callout in the narrative
- [x] 6.3 Include reliability classification: High (>=80%), Moderate (50-79%), Low (<50%) based on `confidence_pct`
- [x] 6.4 Include top 1-3 cost-driver fields identified from `pm_input` values
- [x] 6.5 Expose `explain` from `explainer/__init__.py`

## 7. UI Wizard Scaffold (`ui/wizard.py`)

- [x] 7.1 Create `WizardApp` class with four step frames: Technical, Team, Business, Risk
- [x] 7.2 Implement "Next" / "Back" navigation between steps with state preservation
- [x] 7.3 Render categorical fields as `CTkOptionMenu` dropdowns
- [x] 7.4 Render numeric fields as `CTkSlider` with a live value label
- [x] 7.5 On final step completion, construct `PMInput` and invoke the provided estimation callback

## 8. UI Charts Scaffold (`ui/charts.py`)

- [x] 8.1 Implement `render_dominance_pie(frame, dominance_dict)` — embeds matplotlib pie chart
- [x] 8.2 Implement `render_cost_bar(frame, result, dominance_dict)` — embeds bar chart of cost distribution
- [x] 8.3 Implement `render_risk_heatmap(frame, pm_input)` — embeds heatmap from risk/uncertainty fields
- [x] 8.4 Implement `render_timeline(frame, pm_input, result)` — embeds Gantt-style timeline

## 9. UI Simulation Panel Scaffold (`ui/simulation.py`)

- [x] 9.1 Create `SimulationPanel` widget that displays the current base estimate
- [x] 9.2 Add field selector (dropdown of PMInput field names) and value input control
- [x] 9.3 Wire "Recalculate" button to call `estimator_bridge.simulate` and update the displayed result
- [x] 9.4 Display `compare` output (delta effort, delta %, direction) alongside the simulated result

## 10. Entry-point Wiring

- [x] 10.1 Add `USE_NEW_UI` env-var flag to `app.py` (default `False`)
- [x] 10.2 When `USE_NEW_UI=1`, launch `ui.wizard.WizardApp` instead of the legacy UI
- [x] 10.3 Verify `USE_NEW_UI=0` still runs the existing customtkinter UI without errors
- [x] 10.4 Verify `USE_NEW_UI=1` opens the new wizard without crashing

## 11. Result Window (`ui/result_window.py`)

- [x] 11.1 Create `ResultWindow(CTkToplevel)` that opens after wizard submit, showing effort_pm + confidence range
- [x] 11.2 Add tabbed chart area using `CTkTabview` with tabs: Dominance, Cost Breakdown, Risk, Timeline
- [x] 11.3 Embed each chart type from `ui/charts.py` into the corresponding tab
- [x] 11.4 Add scrollable explainer text area showing output from `explainer.explain()`
- [x] 11.5 Add "Simulate" button that embeds or opens `SimulationPanel` in the result window
- [x] 11.6 Wire `WizardApp.on_complete` callback to open `ResultWindow` instead of printing to console

## 12. Cost Calculator (`core/cost_calculator.py`)

- [x] 12.1 Define `CostResult` dataclass: `total_cost`, `team_needed`, `cost_per_developer`, `currency`, `formatted_total`
- [x] 12.2 Implement `compute(effort_pm, salary_per_month, deadline_months, num_developers, currency="IDR") -> CostResult`
- [x] 12.3 Implement IDR formatting in `formatted_total` (e.g., `"Rp 270,000,000"`)
- [x] 12.4 Expose `CostResult` and `cost_calculator` from `core/__init__.py`

## 13. Accuracy Reporter (`core/accuracy_reporter.py`)

- [x] 13.1 Implement `summarize(result: BridgeResult) -> dict` returning model quality metrics for PM display
- [x] 13.2 Add `quality_label`: Excellent (mmre<=0.10), Good (0.10<mmre<=0.25), Fair (mmre>0.25)
- [x] 13.3 Add `warn=True` and `warn_message` when mmre > 0.15 (SRS MAPE threshold)
- [x] 13.4 Expose `accuracy_reporter` from `core/__init__.py`

## 14. Decision Support (`core/decision_support.py`)

- [x] 14.1 Implement `detect_overrun_risk(pm_input, result, budget_pm) -> dict` with `risk_level` (low/medium/high), `overrun_pm`, `overrun_pct`
- [x] 14.2 Implement `suggest_optimizations(pm_input, result, target_pm) -> list[dict]` returning sorted suggestions with `field`, `suggested_value`, `estimated_saving_pm`, `description`
- [x] 14.3 Implement `advise_risks(pm_input) -> list[dict]` returning mitigation recommendations for high-risk fields
- [x] 14.4 Expose `decision_support` from `core/__init__.py`

## 15. Past Project Matching (`core/past_project.py`)

- [x] 15.1 Implement `save(pm_input, result, confidence, label="")` that appends a record to `last_inputs/history.json`
- [x] 15.2 Implement `load_all() -> list[dict]` that reads all records from `last_inputs/history.json`
- [x] 15.3 Implement `find_similar(pm_input, top_n=3) -> list[dict]` using cosine similarity on numeric PMInput fields
- [x] 15.4 Expose `past_project` from `core/__init__.py`

## 16. Wizard Cost & Budget Step (`ui/wizard.py`)

- [x] 16.1 Add "Cost & Budget" as step 5 in `_STEPS` with `salary_per_month` (slider: 500k–20M IDR, step 500k, default 5M) and `initial_budget` (slider: 10M–2B IDR, step 10M, default 200M)
- [x] 16.2 Add `_SLIDER_FORMATTERS` dict mapping `salary_per_month` and `initial_budget` to IDR currency display functions
- [x] 16.3 Update `_render_step()` to use `_SLIDER_FORMATTERS` for value label text and widen label for currency fields
- [x] 16.4 Change `_submit()` to call `on_complete(pm_input, salary_per_month, initial_budget)` instead of `on_complete(pm_input)`
- [x] 16.5 Fix wizard title encoding: replace garbled em-dash with ASCII `--`

## 17. Cost Display in Result Window (`ui/result_window.py`)

- [x] 17.1 Update `ResultWindow.__init__` signature to accept `salary_per_month: float` and `initial_budget: float`
- [x] 17.2 Call `cost_calculator.compute()` in `__init__` and store as `self._cost`
- [x] 17.3 Add cost summary row below the header: formatted total cost, team_needed, and budget surplus/deficit with green (surplus) or red (deficit) colour

## 18. Past Project Comparison Panel (`ui/result_window.py`)

- [x] 18.1 Call `past_project.find_similar(pm_input, top_n=3)` after saving and store as `self._similar`
- [x] 18.2 Add "History" tab to the CTkTabview; if `self._similar` is empty show "First estimation -- no past projects to compare"; otherwise render a comparison row per similar project (timestamp, effort_pm, confidence_pct, similarity score vs current)
