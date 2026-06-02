## ADDED Requirements

### Requirement: Result window displayed after estimation
`ui/result_window.py` SHALL provide a `ResultWindow` (CTkToplevel) that opens automatically after the wizard completes, displaying the full estimation result including effort, confidence, dominance, explanation, and charts.

#### Scenario: Result window opens after wizard submit
- **WHEN** the wizard's `on_complete` callback is invoked with a valid `PMInput`
- **THEN** a `ResultWindow` opens showing the estimated effort in person-months

#### Scenario: Result window shows confidence range
- **WHEN** the result window is rendered
- **THEN** the effort value is displayed with its lower-upper confidence range and confidence_pct

### Requirement: Result window renders all four charts
The result window SHALL embed all four chart types from `ui/charts.py` in a tabbed or grid layout.

#### Scenario: Dominance pie chart visible
- **WHEN** the result window is open
- **THEN** the dominance pie chart is visible without scrolling

#### Scenario: All chart tabs accessible
- **WHEN** the result window has a tab or panel for each chart type
- **THEN** the PM can navigate to any chart (dominance pie, cost bar, risk heatmap, timeline) without closing the window

### Requirement: Result window shows explainer text
The result window SHALL display the explanation narrative from `explainer.explain()` in a scrollable text area.

#### Scenario: Explanation visible in result window
- **WHEN** the result window opens
- **THEN** the explanation text is visible, including reliability label and top cost drivers

### Requirement: Simulate button in result window
The result window SHALL include a "Simulate" button that opens or embeds the `SimulationPanel` so the PM can run what-if scenarios without leaving the result view.

#### Scenario: Simulate button opens simulation panel
- **WHEN** the PM clicks "Simulate" in the result window
- **THEN** the `SimulationPanel` is shown within or alongside the result window
