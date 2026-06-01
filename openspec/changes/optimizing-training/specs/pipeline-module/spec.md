## MODIFIED Requirements

### Requirement: Merged build script
`pipeline/build.py` SHALL menggabungkan fungsionalitas `auto_configure.py` dan `model_pipeline.py` dan mengekspos tiga entry point: `configure()`, `train()`, dan `build_all()`. Fungsi `train()` SHALL mendukung eksekusi paralel antar dataset maupun antar model dalam satu dataset, dengan default worker count otomatis berdasarkan `os.cpu_count()`.

#### Scenario: Konfigurasi saja
- **WHEN** pengguna menjalankan `python pipeline/build.py --configure`
- **THEN** sistem menghasilkan `dataset_config.yaml` tanpa melatih model

#### Scenario: Training saja (default workers)
- **WHEN** pengguna menjalankan `python pipeline/build.py --train`
- **THEN** sistem melatih model menggunakan konfigurasi yang ada dengan worker count otomatis

#### Scenario: Training dengan worker count eksplisit
- **WHEN** pengguna menjalankan `python pipeline/build.py --train --dataset-workers 1 --model-workers 1`
- **THEN** sistem melatih model secara sekuensial (identik dengan perilaku sebelumnya)

#### Scenario: Full pipeline
- **WHEN** pengguna menjalankan `python pipeline/build.py`
- **THEN** sistem menjalankan configure lalu train
