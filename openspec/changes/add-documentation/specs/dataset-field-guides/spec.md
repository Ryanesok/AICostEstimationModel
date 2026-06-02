## ADDED Requirements

### Requirement: docs/ folder contains one field guide per dataset
A `docs/` folder SHALL exist at the project root containing one Markdown file per supported dataset: `cocomo-81.md`, `desharnais.md`, `china.md`, `maxwell.md`.

#### Scenario: All four guide files exist
- **WHEN** a user opens the `docs/` folder
- **THEN** the folder SHALL contain exactly the files `cocomo-81.md`, `desharnais.md`, `china.md`, and `maxwell.md`

#### Scenario: Guide contains a field reference table
- **WHEN** a user opens any dataset guide
- **THEN** the guide SHALL contain a table with columns: Field, Label, Unit / Scale, Accepted Values, and Example
- **THEN** every input feature listed in `dataset_config.yaml` for that dataset SHALL have a corresponding row in the table

#### Scenario: Guide includes dataset overview section
- **WHEN** a user opens any dataset guide
- **THEN** the guide SHALL begin with a brief overview section describing the dataset origin, domain, and the effort unit used (person-months or person-hours)

#### Scenario: Guide notes effort unit and output conversion
- **WHEN** a user reads a guide for a person-hours dataset (Desharnais, China, Maxwell)
- **THEN** the guide SHALL note that raw model output is in person-hours and that the app divides by 160 to display person-months
