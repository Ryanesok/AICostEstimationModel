## ADDED Requirements

### Requirement: Hybrid model CV evaluation
`pipeline/build.py` SHALL menghitung performa CV untuk model Hybrid (weighted ensemble dari SVR, LR, RF, XGB, LSTM) selama training dan menyimpan metrik `hybrid_cv_mae_mean`, `hybrid_mmre`, `hybrid_pred25` ke `models/*_metrics.json`.

#### Scenario: Hybrid metrics tersimpan setelah training
- **WHEN** training selesai untuk suatu dataset
- **THEN** file `models/<stem>_metrics.json` mengandung field `hybrid_cv_mae_mean`, `hybrid_mmre`, `hybrid_pred25` dengan nilai numerik

#### Scenario: Hybrid weights mencakup semua 5 model
- **WHEN** training selesai
- **THEN** `hybrid_weights` di metrics JSON mengandung key `svr`, `lr`, `rf`, `xgb`, `lstm` dan jumlah seluruh bobot = 1.0

### Requirement: Hybrid model dapat dipilih sebagai best estimator
`pipeline/estimator.py` SHALL mendaftarkan "Hybrid" sebagai kandidat model di `select_best_estimator`, sehingga jika pred25 hybrid lebih baik dari semua model tunggal, Hybrid dipilih.

#### Scenario: Hybrid bersaing dalam seleksi
- **WHEN** `select_best_estimator` dipanggil dan metrics JSON punya `hybrid_cv_mae_mean`
- **THEN** Hybrid dimasukkan sebagai kandidat dan dibandingkan dengan model tunggal lain berdasarkan MAE dan PRED25

#### Scenario: Hybrid dipilih jika terbaik
- **WHEN** Hybrid memiliki `cv_mae_pm` terendah di antara semua kandidat
- **THEN** `select_best_estimator` mengembalikan `EstimatorResult` dengan `model_name == "Hybrid"`

### Requirement: run_hybrid_estimate mendukung SVR dan RF
`pipeline/estimator.py` `run_hybrid_estimate` SHALL membaca bobot `svr` dan `rf` dari `hybrid_weights` dan menggunakannya jika model tersedia.

#### Scenario: SVR berkontribusi ke prediksi hybrid
- **WHEN** `hybrid_weights` mengandung key `svr` dan model SVR tersedia
- **THEN** prediksi SVR dimasukkan ke weighted average dengan bobot yang sesuai

#### Scenario: RF berkontribusi ke prediksi hybrid
- **WHEN** `hybrid_weights` mengandung key `rf` dan model RF tersedia
- **THEN** prediksi RF dimasukkan ke weighted average dengan bobot yang sesuai
