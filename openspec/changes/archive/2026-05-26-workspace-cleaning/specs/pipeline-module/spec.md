## ADDED Requirements

### Requirement: Pipeline package structure
Semua skrip non-UI SHALL ditempatkan di dalam folder `pipeline/` sebagai Python package dengan file `__init__.py`.

#### Scenario: Package dapat diimport
- **WHEN** pengguna menjalankan `from pipeline.estimator import load_estimator_models` dari root project
- **THEN** import berhasil tanpa error `ModuleNotFoundError`

#### Scenario: Root directory bersih
- **WHEN** pengguna melihat isi root directory
- **THEN** hanya terdapat `app.py`, `README.md`, `requirements.txt`, dan folder-folder (`pipeline/`, `models/`, `data/`, `last_inputs/`, `.claude/`, `.vscode/`, `openspec/`)

### Requirement: Merged build script
`pipeline/build.py` SHALL menggabungkan fungsionalitas `auto_configure.py` dan `model_pipeline.py` dan mengekspos tiga entry point: `configure()`, `train()`, dan `build_all()`.

#### Scenario: Konfigurasi saja
- **WHEN** pengguna menjalankan `python pipeline/build.py --configure`
- **THEN** sistem menghasilkan `dataset_config.yaml` tanpa melatih model

#### Scenario: Training saja
- **WHEN** pengguna menjalankan `python pipeline/build.py --train`
- **THEN** sistem melatih model menggunakan konfigurasi yang sudah ada

#### Scenario: Full pipeline
- **WHEN** pengguna menjalankan `python pipeline/build.py`
- **THEN** sistem menjalankan configure lalu train secara berurutan (setara dengan urutan lama)

### Requirement: Config files dalam pipeline/
File `dataset_config.yaml`, `field_labels.yaml`, dan `datasets.txt` SHALL berada di dalam `pipeline/`.

#### Scenario: app.py membaca config dari pipeline/
- **WHEN** `app.py` dipanggil dan mencari `dataset_config.yaml`
- **THEN** file dibaca dari path `pipeline/dataset_config.yaml` dan aplikasi berjalan normal

### Requirement: app.py import dari pipeline package
`app.py` SHALL mengimport dari `pipeline.estimator` sebagai pengganti `estimator`.

#### Scenario: Aplikasi dapat dijalankan
- **WHEN** pengguna menjalankan `python app.py`
- **THEN** aplikasi GUI terbuka tanpa `ImportError`
