## Context

`model_pipeline.py::preprocess()` saat ini mengembalikan `(X_scaled, y, scaler)` di mana `y` adalah raw effort values. Semua model (SVR, LR, RF, XGB, LSTM) dilatih langsung pada `y` tersebut. Data effort bersifat log-normal: misalnya dataset nasa93 memiliki effort berkisar 6–11.400 person-months, desharnais 69–9.807 person-hours. Skewness yang tinggi menyebabkan MSE dan MAE didominasi oleh outlier besar, sehingga model underfit pada proyek kecil-menengah.

`estimator.py::run_estimate()` mengambil raw output model dan mengonversinya ke person-months via `_to_person_months()`. Saat ini tidak ada lapisan inverse-transform.

## Goals / Non-Goals

**Goals:**
- Terapkan `log1p` transform pada `y` sebelum training jika flag dataset aktif
- Terapkan `expm1` inverse-transform pada output prediksi di `run_estimate()` dan `run_hybrid_estimate()`
- Auto-deteksi skewness di `auto_configure.py` dan set flag default
- Semua metrik (MAE, RMSE, MMRE, PRED25) dilaporkan dalam skala asli
- Backward-compatible: dataset lama tanpa flag `log_transform_target` di config berperilaku sama seperti sebelumnya (tidak ada transform)

**Non-Goals:**
- Normalisasi target dengan MinMax atau StandardScaler (log1p sudah cukup untuk data log-normal)
- Transformasi Box-Cox atau Yeo-Johnson (lebih kompleks, tidak dibutuhkan)
- Ubah normalisasi fitur (sudah memadai dengan StandardScaler)
- Normalisasi per-model yang berbeda (satu flag per dataset berlaku untuk semua model)

## Decisions

### 1. `np.log1p` bukan `np.log`
**Pilihan**: `log1p(y)` (= `log(y+1)`) bukan `log(y)`.  
**Alasan**: Beberapa dataset mungkin mengandung nilai target 0 (proyek trivial). `log(0)` = `-inf`; `log1p(0)` = `0`. Aman tanpa perlu filter tambahan.  
**Alternatif ditolak**: `log(y + eps)` dengan epsilon kecil — tidak intuitif dan sulit di-inverse.

### 2. Flag per-dataset di `dataset_config.yaml`, bukan global
**Pilihan**: `log_transform_target: true/false` di setiap entri dataset.  
**Alasan**: Tidak semua dataset memiliki skewness tinggi. Dataset yang datanya sudah mendekati normal tidak perlu transform. Flag memungkinkan override manual.  
**Alternatif ditolak**: Transform global — akan merusak dataset dengan distribusi target yang sudah baik.

### 3. Auto-deteksi skewness di `auto_configure.py`
**Pilihan**: Hitung `scipy.stats.skew(y)` atau `pandas.Series.skew()` pada kolom target. Jika skewness > 1.0, set `log_transform_target: true`.  
**Alasan**: Threshold 1.0 adalah patokan umum (moderate-to-high skew). Untuk dataset yang sudah dikonfigurasi, flag TIDAK diubah otomatis — hanya saat pertama kali dikonfigurasi.  
**Alternatif ditolak**: Selalu `true` untuk semua dataset — tidak fleksibel; bisa memperburuk dataset dengan distribusi normal.

### 4. Metrik dihitung dalam skala asli
**Pilihan**: Sebelum menghitung MAE/RMSE/MMRE/PRED25 dalam CV loop, terapkan `np.expm1()` pada `y_pred` dan `y_true` (jika flag aktif).  
**Alasan**: Metrik dalam log-space tidak dapat diinterpretasi oleh pengguna ("MAE = 2.3 log-hours" tidak bermakna).  
**Alternatif ditolak**: Simpan metrik dalam log-space — membingungkan.

### 5. Simpan flag ke `_metrics.json` dan load di `estimator.py`
**Pilihan**: `evaluate_models()` menyimpan `log_transform_target: true/false` ke metrics dict. `load_estimator_models()` membaca flag dari metrics file dan menyimpannya di `LoadedModels`. `run_estimate()` menggunakan flag untuk inverse-transform.  
**Alasan**: `run_estimate()` perlu tahu apakah model dilatih di log-space tanpa membaca `dataset_config.yaml` ulang (decoupled dari config).

## Risks / Trade-offs

- **Retrain wajib**: Semua model yang sudah ada harus di-retrain setelah implementasi. Model lama di-retrain akan menghasilkan metrik berbeda (seharusnya lebih baik). → Komunikasikan ke user bahwa perlu `python model_pipeline.py` setelah merge.
- **Backward-compat dataset lama**: Dataset yang sudah ada di config tanpa flag `log_transform_target` akan default ke `false` via `.get("log_transform_target", False)` — perilaku tidak berubah sampai flag ditambahkan/dideteksi. → Jalankan `auto_configure.py` atau tambahkan flag manual ke config untuk mengaktifkan transform pada dataset lama.
- **LSTM loss scale**: LSTM dilatih dengan MSE loss. Setelah log-transform, MSE dalam log-space jauh lebih kecil — learning rate yang sama mungkin perlu divalidasi. → Hyperparameter tuning yang ada (5-fold CV) sudah menyeleksi `lr` terbaik, sehingga ini self-correcting.
