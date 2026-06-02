## MODIFIED Requirements

### Requirement: Merged build script
`pipeline/build.py` SHALL menggabungkan fungsionalitas `auto_configure.py` dan `model_pipeline.py` dan mengekspos tiga entry point: `configure()`, `train()`, dan `build_all()`. `evaluate_models()` SHALL mengembalikan metrik hybrid (MAE, MMRE, PRED25) bersama metrik model tunggal, dengan mengumpulkan out-of-fold predictions dari setiap model dan menghitung weighted average menggunakan inverse-MAE weights dari SVR, LR, RF, XGB, dan LSTM.

#### Scenario: Konfigurasi saja
- **WHEN** pengguna menjalankan `python pipeline/build.py --configure`
- **THEN** sistem menghasilkan `dataset_config.yaml` tanpa melatih model

#### Scenario: Training saja
- **WHEN** pengguna menjalankan `python pipeline/build.py --train`
- **THEN** sistem melatih model menggunakan konfigurasi yang sudah ada

#### Scenario: Full pipeline
- **WHEN** pengguna menjalankan `python pipeline/build.py`
- **THEN** sistem menjalankan configure lalu train secara berurutan (setara dengan urutan lama)

#### Scenario: Hybrid metrics dicetak saat training
- **WHEN** training selesai untuk suatu dataset
- **THEN** stdout menampilkan baris berformat `[<stem>] Hybrid CV-MAE=X.XX  PRED25=X.XX  weights=svr=X.XXX lr=X.XXX rf=X.XXX xgb=X.XXX lstm=X.XXX`
