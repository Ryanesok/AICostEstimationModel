## Context

`app.py` currently uses a 2-column CustomTkinter grid: column 0 is a `CTkScrollableFrame` (460px wide) containing the dataset selector and all input fields; column 1 is a `CTkFrame` containing the matplotlib chart. The model toggle, results area, and calculator all live inside the scrollable left panel, requiring the user to scroll to see them while the chart is visible.

The window is fixed at 1000×780 with `resizable(False, False)`.

## Goals / Non-Goals

**Goals:**
- 3-column window at 1400×780 (fixed)
- Left (≈420px): dataset selector + scrollable input form
- Middle (≈380px): model toggle, result display, cost/team calculator — always fully visible, no scrolling
- Right (≈480px): chart — unchanged content, same code
- All existing logic (model loading, YAML reading, prediction, chart drawing) untouched

**Non-Goals:**
- Color scheme, fonts, or widget style changes
- Making the window resizable
- Changing any business logic or calculator math

## Decisions

**Decision: Extract middle column as a dedicated `CTkFrame`**

Currently the model toggle, result, and calculator widgets are placed inside `_build_form_area()` which puts everything into the scrollable left frame. Moving them out into a fixed middle column requires splitting `_build_form_area()` into two methods:
- `_build_form_area(parent)` — only builds dataset selector + input fields (left column)
- `_build_middle_panel(parent)` — builds model toggle, result area, and calculator (middle column)

This is the cleanest split: each method owns one column, reducing coupling.

**Decision: Middle column is a plain `CTkFrame` (not scrollable)**

The middle column contains a fixed set of widgets (model toggle, 2 result labels, 4 calculator rows). Its height is predictable and fits within 780px without scrolling.

**Decision: Keep `_build_ui()` as the single grid setup point**

```
grid_columnconfigure(0, weight=0, minsize=420)   # left: form
grid_columnconfigure(1, weight=0, minsize=380)   # middle: result+calc
grid_columnconfigure(2, weight=1)                # right: chart
grid_rowconfigure(0, weight=1)
```

Using `weight=1` on only the chart column means extra horizontal space (if any) goes to the chart, which is the least content-dense panel.

## Risks / Trade-offs

- [Risk] Middle column height overflows on very long datasets (many fields) → Mitigation: middle column content is independent of dataset; it has a fixed number of widgets regardless of dataset choice.
- [Risk] Tight horizontal fit on lower-resolution screens → Mitigation: window is fixed at 1400px; targeting 1080p+ displays. Same policy as the current 1000px fixed window.

## Migration Plan

1. Update `self.geometry("1400x780")` in `__init__`
2. Rewrite `_build_ui()` grid configuration for 3 columns
3. Extract middle widgets from `_build_form_area()` into new `_build_middle_panel(middle_frame)` method
4. Adjust `_build_form_area()` to only build left-column content
5. Update any `row=` indices affected by the removal of middle widgets from the left frame

No rollback needed — this is a pure UI restructure with no data or model changes.
