## Why

The project has no onboarding documentation — a new user opening the app has no way to know what each dataset field means, what valid input ranges are, or how to run the pipeline. A README and per-dataset field guides close this gap and make the tool usable without reading the source code.

## What Changes

- Add `README.md` at the project root with a quick-start guide covering setup, running the pipeline, and launching the app
- Add `docs/` folder with one Markdown guide per dataset (`cocomo-81.md`, `desharnais.md`, `china.md`, `maxwell.md`), each containing a table of all fields with accepted values, units, and example inputs
- Add `roadmap.md` at the project root summarising completed work and the planned direction so contributors can see where the project is headed

## Capabilities

### New Capabilities
- `root-readme`: A root-level README covering project purpose, prerequisites, setup steps (install, configure, train, launch), and file layout
- `dataset-field-guides`: One Markdown document per dataset inside `docs/`, each explaining every input field — label, description, unit, valid range or accepted values, and a concrete example
- `project-roadmap`: A root-level `roadmap.md` listing what has shipped, what is in progress, and the planned next steps

### Modified Capabilities
<!-- none -->

## Impact

- New files only — no existing code is changed
- `docs/cocomo-81.md`, `docs/desharnais.md`, `docs/china.md`, `docs/maxwell.md`
- `README.md` at project root
- `roadmap.md` at project root
- Source of truth for field content: `field_labels.yaml` (labels + hints) and `dataset_config.yaml` (features list, effort_unit, target)
