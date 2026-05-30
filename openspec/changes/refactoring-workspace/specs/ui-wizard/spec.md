## ADDED Requirements

### Requirement: Wizard package scaffold
The `ui/` directory SHALL exist as a Python package with `__init__.py`, containing `wizard.py`, `charts.py`, and `simulation.py`.

#### Scenario: ui package importable
- **WHEN** `import ui` is executed from the project root
- **THEN** no `ImportError` is raised

### Requirement: Step-by-step input wizard
`ui/wizard.py` SHALL implement a multi-step wizard that collects PM inputs one dimension at a time (Technical → Team → Business → Risk) and returns a fully-populated `PMInput` on completion.

#### Scenario: Wizard advances to next step
- **WHEN** the PM fills all fields on the current step and clicks "Next"
- **THEN** the wizard displays the next dimension's fields

#### Scenario: Wizard allows back navigation
- **WHEN** the PM is on step 2 or later and clicks "Back"
- **THEN** the wizard returns to the previous step with the previously entered values preserved

#### Scenario: Wizard produces PMInput on completion
- **WHEN** the PM completes all four steps and clicks "Estimate"
- **THEN** a `PMInput` instance is constructed and passed to the estimation callback

#### Scenario: Wizard completes within 5–10 minutes
- **WHEN** a PM with no prior experience fills in all fields using dropdowns and sliders
- **THEN** the total time to reach the Estimate button does not require more than 10 minutes (enforced by limiting fields per step to ≤6)

### Requirement: Input controls — dropdowns and sliders only
Each wizard step SHALL use dropdowns (`CTkOptionMenu`) for categorical fields and sliders (`CTkSlider`) for numeric fields. Free-text input SHALL NOT be used for PM-facing parameters.

#### Scenario: Categorical field uses dropdown
- **WHEN** the wizard renders `feature_complexity`
- **THEN** a dropdown with options ["low", "medium", "high"] is displayed

#### Scenario: Numeric field uses slider with visible current value
- **WHEN** the wizard renders `num_features`
- **THEN** a slider is displayed with a label showing the current integer value

### Requirement: Charts panel
`ui/charts.py` SHALL provide rendering functions for: effort pie chart (dominance), cost distribution bar chart, timeline estimation Gantt-style view, and risk heatmap.

#### Scenario: Pie chart renders dominance data
- **WHEN** `charts.render_dominance_pie(frame, dominance_dict)` is called
- **THEN** a matplotlib pie chart is embedded in `frame` showing each dimension's share

#### Scenario: Risk heatmap renders without error
- **WHEN** `charts.render_risk_heatmap(frame, pm_input)` is called
- **THEN** a heatmap widget is embedded in `frame` without raising an exception
