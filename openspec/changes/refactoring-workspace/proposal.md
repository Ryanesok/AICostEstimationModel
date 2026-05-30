## Why

The current codebase has a working ML pipeline and a monolithic `app.py` (35KB, customtkinter GUI) that exposes dataset-specific technical fields — but the SRS defines a much richer product: PM-facing wizard UX, confidence levels, dominance breakdown, scenario simulation, and decision support. The workspace must be restructured now — before adding any new features — so the architecture can scale to all SRS requirements without accumulating debt in a single oversized file.

## What Changes

- **Decompose `app.py`** into focused modules: `ui/`, `core/`, `explainer/` — current file is too large to extend safely
- **Replace raw dataset fields** with the PM-friendly parameter schema defined in `project_parameter.txt` (technical, team, business, risk dimensions)
- **Add a `ui/` package** under the app layer for wizard-step UX, charts, and simulation panels — separate from the ML pipeline
- **Add a `core/` package** for estimation orchestration, confidence calculation, and dominance analysis
- **Add an `explainer/` module** for output explanation (why this result, what drives cost, reliability)
- **Introduce a `config/` or `schemas/` layer** for PM-input schema definitions, keeping field labels and parameter mappings out of YAML config files that currently serve the ML pipeline

## Capabilities

### New Capabilities

- `pm-input-schema`: Structured input parameters for PM (technical, team, business, risk) as defined in `project_parameter.txt`
- `ui-wizard`: Step-by-step wizard UX with dropdowns, sliders, and checklists replacing the current form-based input
- `estimation-core`: Orchestration layer that takes PM inputs, maps to ML features, runs estimation, and returns structured results
- `output-explainer`: Explains results — why this estimate, which factors dominate, confidence level with range
- `scenario-simulator`: Recalculates estimate when PM changes a single variable (e.g., "add 2 developers")

- `ui-result-window`: Full result display window after wizard — effort, confidence, all charts, explainer text, and simulation panel in one view
- `cost-calculator`: Converts person-months to total project cost (IDR) and team sizing recommendation
- `accuracy-reporter`: Surfaces model quality metrics (MAPE, MAE, pred25) to PM with quality label and SRS-threshold warning
- `decision-support`: Cost overrun detector, resource optimization suggestions, risk mitigation advisor
- `past-project-match`: Persist and retrieve past estimates; cosine-similarity lookup of similar historical projects

### Modified Capabilities

- `pipeline-module`: No requirement changes; internal implementation may be touched during decomposition but spec-level behavior is unchanged

## Impact

- `app.py`: Will be reduced to an entry-point shim; logic moves into `ui/`, `core/`, `explainer/`
- `pipeline/estimator.py`: Remains the ML backend; `core/` will call it
- `pipeline/field_labels.yaml` + `dataset_config.yaml`: Remain for the ML pipeline; new PM parameter schema lives separately
- No breaking changes to the ML training pipeline (`pipeline/build.py`)
