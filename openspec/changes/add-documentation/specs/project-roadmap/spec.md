## ADDED Requirements

### Requirement: Root roadmap describes project direction
A `roadmap.md` SHALL exist at the project root and outline the project's development direction across three states: shipped, in progress, and planned.

#### Scenario: Roadmap present at root
- **WHEN** a contributor opens the project directory
- **THEN** a `roadmap.md` file SHALL be present at the root

#### Scenario: Roadmap separates shipped, in-progress, and planned work
- **WHEN** a reader opens `roadmap.md`
- **THEN** the file SHALL contain three clearly labelled sections — completed/shipped milestones, work currently in progress, and planned future direction

#### Scenario: Roadmap is reachable from the README
- **WHEN** a reader follows the README
- **THEN** the README SHALL link to `roadmap.md` so the roadmap is discoverable without searching the file tree
