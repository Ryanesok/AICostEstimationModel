## ADDED Requirements

### Requirement: Root README provides quick-start guide
A `README.md` SHALL exist at the project root and cover: project purpose, prerequisites, setup commands, usage workflow, and file layout.

#### Scenario: README present at root
- **WHEN** a user opens the project directory
- **THEN** a `README.md` file SHALL be present at the root

#### Scenario: README covers prerequisites and setup
- **WHEN** a reader follows the README
- **THEN** the README SHALL list Python version requirement, `pip install -r requirements.txt` command, and the four-step workflow: `downloader.py` → `auto_configure.py` → `model_pipeline.py` → `app.py`

#### Scenario: README describes file layout
- **WHEN** a reader consults the README
- **THEN** the README SHALL include a file/folder layout section identifying key files (`app.py`, `model_pipeline.py`, `auto_configure.py`, `data/`, `models/`, `docs/`, `field_labels.yaml`, `dataset_config.yaml`)
