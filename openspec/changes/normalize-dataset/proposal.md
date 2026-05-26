## Why

Fitur (`X`) sudah dinormalisasi dengan `StandardScaler`, tetapi **target variable (`y` — effort)** belum pernah ditransformasi. Data effort bersifat *right-skewed* (log-normal): nilainya bisa berkisar dari puluhan hingga ribuan person-hours/months dalam satu dataset. Model yang dilatih pada nilai raw mendominasi error-nya pada proyek besar, sehingga MAE dan MMRE menjadi lebih tinggi dari yang seharusnya. Log-transform target (`np.log1p`) sebelum training lalu inverse-transform (`np.expm1`) saat prediksi adalah praktik standar dalam literatur Software Effort Estimation (SEEE).

## What Changes

- Tambah flag `log_transform_target: true/false` per dataset di `dataset_config.yaml`; default `true` jika *skewness* target > 1.0 (dideteksi otomatis oleh `auto_configure.py`)
- Di `preprocess()` (`model_pipeline.py`): jika flag aktif, terapkan `np.log1p(y)` sebelum training; kembalikan flag sebagai bagian dari output
- Semua fungsi training menerima dan melatih model pada log-space target
- Simpan flag `log_transform_target` ke `_metrics.json` per dataset
- Di `run_estimate()` / `run_hybrid_estimate()` (`estimator.py`): jika flag aktif, terapkan `np.expm1()` pada output prediksi agar hasilnya kembali ke satuan asli (person-hours/months)
- Di `auto_configure.py`: hitung skewness target, sarankan/set `log_transform_target` secara otomatis
- Metrik (MAE, RMSE, MMRE, PRED25) tetap dihitung dalam **skala asli** (setelah inverse-transform) agar dapat diinterpretasi

## Capabilities

### New Capabilities

- `target-log-transform`: Transformasi log1p pada target variable sebelum training, dengan inverse-transform saat prediksi; termasuk auto-deteksi skewness di `auto_configure.py` dan persistensi flag ke config + metrics JSON

### Modified Capabilities

*(none — perubahan pada `preprocess()` dan `run_estimate()` adalah detail implementasi, bukan perubahan requirement yang sudah terdokumentasi di spec)*

## Impact

- `model_pipeline.py` — fungsi `preprocess()`, `evaluate_models()`: menerapkan dan meneruskan flag log-transform
- `estimator.py` — fungsi `run_estimate()`, `run_hybrid_estimate()`: inverse-transform output jika flag aktif
- `auto_configure.py` — menambah deteksi skewness dan penulisan `log_transform_target` ke config
- `dataset_config.yaml` — tambah field `log_transform_target` per dataset
- `models/<dataset>_metrics.json` — tambah field `log_transform_target` (bool)
- Model yang sudah ada **harus di-retrain** setelah perubahan ini diterapkan
