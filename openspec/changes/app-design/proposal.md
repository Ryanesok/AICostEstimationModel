## Why

The current app uses a 2-column layout (scrollable form on the left, chart on the right) that makes the model toggle, result display, and calculator feel cramped and visually disconnected from both the input form and the chart. A 3-column layout separates concerns clearly and allows a wider window to present all information without scrolling.

## What Changes

- Window width increases from 1000px to ~1400px; height stays fixed at 780px
- Left column (≈420px): dataset selector + scrollable input form fields
- Middle column (≈400px): model toggle (SVR/LR), estimation result display, and cost/team calculator
- Right column (≈480px): bar chart visualization
- Left and right columns remain scrollable/fixed as needed; middle column is always fully visible
- No changes to color scheme, fonts, or branding

## Capabilities

### New Capabilities

- `three-column-layout`: 3-panel fixed layout replacing the current 2-panel split; left=form, middle=results+calculator, right=chart

### Modified Capabilities

- none

## Impact

- `app.py`: entire `_build_ui()`, `_build_form_area()`, and grid configuration rewritten; widget placement updated throughout
- Window geometry string updated
- No changes to model loading, prediction logic, YAML config reading, or calculator math
