## Context

The project is a desktop cost-estimation tool with four datasets (COCOMO-81, Desharnais, China, Maxwell). Field metadata (labels, hints, valid values) already exists in `field_labels.yaml` and `dataset_config.yaml`. No documentation exists for end users — there is no README and no reference guide explaining what to enter into each form field.

## Goals / Non-Goals

**Goals:**
- One `README.md` at the project root covering purpose, prerequisites, setup, and usage
- One Markdown file per dataset inside `docs/` — readable without opening the app
- Field tables derived from existing `field_labels.yaml` data so the content stays accurate

**Non-Goals:**
- Auto-generated docs (no tooling to generate from YAML at runtime)
- API or developer reference documentation
- Translations — docs are written in Indonesian to match the existing UI language

## Decisions

**D1 — Static Markdown, not generated**
Option A: Hand-written Markdown files in `docs/`. Option B: Script that generates them from `field_labels.yaml` at build time.
→ Chose A. The dataset list is fixed (4 datasets). Manual authoring produces richer explanations (context on COCOMO effort multipliers, FP counting rules). A generator would need maintenance and adds a build step.

**D2 — One file per dataset, not one big file**
Keeps each guide focused. Users opening a guide for Desharnais don't need to scroll past COCOMO-81's 16 fields.

**D3 — Field table format**
Each field row: `| Field | Label | Unit / Scale | Accepted Values | Example |`
This maps directly to the hint text already in `field_labels.yaml`, giving consistent coverage with no gaps.

**D4 — README scope**
README covers: what the project does, prerequisites (Python 3.11+, pip), four commands (download → configure → train → run), and a file layout section. No architecture deep-dive — that belongs in the design docs already in `openspec/`.

## Risks / Trade-offs

- [Content drift] `field_labels.yaml` is editable; if it is updated, the docs become stale. → Mitigation: README notes that `field_labels.yaml` is authoritative; docs are a human-friendly view of it.
- [Chinese/Maxwell field meaning ambiguity] Some Maxwell T-factor descriptions are terse. → Mitigation: expand from standard Maxwell dataset paper descriptions.

## Migration Plan

No migration required — pure additions. No existing code is changed.
