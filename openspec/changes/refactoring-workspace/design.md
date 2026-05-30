## Context

The project is a Python desktop app (customtkinter + matplotlib) with an ML estimation pipeline. Current state:
- `app.py` (~35KB): monolithic UI + orchestration + chart rendering, all in one file
- `pipeline/`: mature ML backend (`estimator.py`, `build.py`, dataset configs)
- No separation between PM-facing input schema and ML-internal feature schema
- No explainability, confidence, dominance, or simulation logic exists yet

The SRS requires five new capabilities (wizard UX, PM input schema, estimation core, output explainer, scenario simulator) that cannot be cleanly added to the current monolith without making it unmaintainable.

## Goals / Non-Goals

**Goals:**
- Split `app.py` into a layered module structure: `ui/`, `core/`, `explainer/`
- Define a PM-input schema (technical / team / business / risk) that maps to ML features
- Keep the `pipeline/` ML backend untouched (no breaking changes)
- Establish the folder scaffold that each new capability can slot into

**Non-Goals:**
- Implementing the wizard UX, charts, or simulation UI (that is `ui-wizard` capability work)
- Training new models or changing ML logic
- Building the full explainer or confidence logic (that is `output-explainer` capability work)
- Migrating existing dataset-based field configs to the new PM schema

## Decisions

### 1 — Three-layer architecture: `ui/` · `core/` · `explainer/`

```
app.py                  ← thin entry point (launch window only)
ui/
  __init__.py
  wizard.py             ← step-by-step PM input wizard
  charts.py             ← pie chart, cost distribution, timeline, heatmap
  simulation.py         ← scenario recalculation panel
core/
  __init__.py
  schema.py             ← PM input dataclass (technical, team, business, risk)
  estimator_bridge.py   ← maps PM params → pipeline features, calls pipeline.estimator
  confidence.py         ← confidence interval and reliability score
  dominance.py          ← per-category effort share (backend %, QA %, etc.)
explainer/
  __init__.py
  result_explainer.py   ← "why this estimate", "what drives cost", reliability narrative
```

**Why this over a single flat module list:** The three layers mirror the SRS concern boundaries — input collection (`ui/`), computation (`core/`), output interpretation (`explainer/`). Each is independently testable and independently extensible.

**Alternative considered:** Keeping everything in `app.py` with more functions — rejected because the file is already 35KB with minimal features; adding five capabilities would exceed 100KB and become impossible to navigate.

### 2 — PM input schema as a Python dataclass in `core/schema.py`

```python
@dataclass
class PMInput:
    # Technical
    num_features: int
    feature_complexity: str        # low / medium / high
    num_apis: int
    platform: str
    third_party_integrations: int
    security_level: str

    # Team
    num_developers: int
    seniority: str                 # junior / mid / senior / mixed
    stack_experience: str

    # Business
    deadline_months: float
    budget_constraint: str         # none / soft / hard
    requirement_change_risk: str   # low / medium / high

    # Risk
    uncertainty_level: str
    technical_debt: str
    external_dependencies: int
```

**Why dataclass over dict:** Type-checked, IDE-friendly, easy to serialize to JSON for logging/replay. The wizard UI constructs one `PMInput`; the bridge maps it to the ML feature vector.

**Alternative considered:** Keeping YAML-driven field config (current approach) — rejected for PM-facing params because YAML configs are designed for dataset columns, not for user-visible wizard steps with business-meaningful labels.

### 3 — `estimator_bridge.py` as the seam between PM schema and ML pipeline

`core/estimator_bridge.py` translates a `PMInput` into the numeric feature vector expected by `pipeline.estimator.run_estimate()`. This keeps the mapping logic in one place and decouples UI from ML internals.

**Why a bridge rather than modifying estimator.py:** `pipeline/estimator.py` is trained-model-aware and mature. Changing its interface risks breaking the build pipeline. A bridge layer absorbs the translation without touching the ML layer.

### 4 — Preserve existing `app.py` behaviour during migration

During the refactoring, the old customtkinter-based UI code stays in `app.py` as a legacy path until each capability is migrated. A `USE_NEW_UI` flag (env var or config key) will gate the new wizard, so the app remains runnable at every commit.

**Why not a hard cutover:** The ML pipeline and dataset-based UI still work; a flag-gated migration lets the team verify each piece before removing the old code.

## Risks / Trade-offs

- **Mapping PM inputs to ML features is lossy** → Mitigation: `estimator_bridge.py` will document each mapping assumption; unmappable params become proxy scores (e.g., `feature_complexity` maps to a synthetic `adjusted_fp` value)
- **customtkinter has limited widget composability for wizard steps** → Mitigation: wizard uses a `CTkTabview` or a frame-stack pattern; complex layout is deferred to the `ui-wizard` capability
- **Parallel old/new code increases maintenance surface temporarily** → Mitigation: each merged PR removes the old counterpart; the flag is removed once all capabilities are migrated
- **dataclass schema may need versioning if PM inputs change** → Mitigation: keep schema in one file; changes are visible in git diff

## Migration Plan

1. Create empty package scaffolds: `ui/__init__.py`, `core/__init__.py`, `explainer/__init__.py`
2. Move `core/schema.py` (PMInput dataclass) — no UI changes yet
3. Add `core/estimator_bridge.py` with stub mapping — app still uses old path
4. Each subsequent capability PR migrates one section of `app.py` into the appropriate package
5. When all sections are migrated, remove the `USE_NEW_UI` flag and delete the old `app.py` blocks

Rollback: any step is reversible by reverting the relevant module addition; `pipeline/` is never modified.

## Open Questions

- **Effort unit output**: Should `core/` return effort in person-months, person-hours, or both? (Current pipeline uses person-months)
- **Confidence interval method**: Bootstrap resampling vs. prediction interval from model residuals — needs decision before `confidence.py` is implemented
- **PM schema validation**: Should invalid inputs (e.g., `num_developers = 0`) raise at the dataclass level or in the bridge?
