## 1. Config & Auto-configure

- [x] 1.1 Tambahkan field `log_transform_target` ke setiap dataset di `dataset_config.yaml` berdasarkan skewness aktual (hitung manual atau jalankan skrip sementara): nasa93, desharnais, china, kitchenham
- [x] 1.2 Di `auto_configure.py`, hitung skewness target (`pd.Series.skew()`) setelah `column_map` diterapkan; set `log_transform_target: True` jika `abs(skewness) > 1.0`, else `False`
- [x] 1.3 Di `auto_configure.py`, pastikan flag TIDAK ditimpa untuk dataset yang sudah ada di config (`if "log_transform_target" not in existing_entry`)

## 2. Preprocessing Pipeline (model_pipeline.py)

- [x] 2.1 Ubah signature `preprocess()` agar menerima `log_transform: bool = False` dan kembalikan `(X_scaled, y, scaler, log_transform)` — `y` di-`log1p` jika flag aktif
- [x] 2.2 Di semua call-site `preprocess()` dalam `evaluate_models()`, baca `log_transform` dari config via `cfg.get("log_transform_target", False)` dan teruskan ke fungsi
- [x] 2.3 Tangkap nilai `log_transform` yang dikembalikan dari `preprocess()` untuk digunakan di langkah metrik

## 3. Metric Computation in Original Scale (model_pipeline.py)

- [x] 3.1 Buat helper `_inverse_transform(arr, log_transform: bool)` yang mengembalikan `np.expm1(arr)` jika flag aktif, else `arr` unchanged
- [x] 3.2 Terapkan `_inverse_transform` pada OOF predictions dan `y` sebelum menghitung MAE, RMSE, MMRE, PRED25 untuk SVR, LR, RF, XGB
- [x] 3.3 Di `_lstm_cv_scores_with_params()`, terapkan `_inverse_transform` pada `all_y_pred` dan `all_y_true` sebelum menghitung semua metrik
- [x] 3.4 Simpan `log_transform_target` (bool) ke dict metrics sebelum ditulis ke `_metrics.json`

## 4. Inference (estimator.py)

- [x] 4.1 Tambah field `log_transform_target: bool = False` ke dataclass `LoadedModels`
- [x] 4.2 Di `load_estimator_models()`, baca `metrics.get("log_transform_target", False)` dan simpan ke `m.log_transform_target`
- [x] 4.3 Di `run_estimate()`, setelah mendapat raw prediction, terapkan `math.expm1(raw)` jika `models.log_transform_target` sebelum `_to_person_months()`
- [x] 4.4 Di `run_hybrid_estimate()`, terapkan inverse-transform pada tiap model output sebelum weighted average (transform dilakukan per model di log-space, lalu di-average, lalu inverse; **atau** inverse dulu lalu average — pilih inverse-then-average agar weighted mean dalam skala asli); dokumentasikan pilihan ini dengan komentar satu baris

## 5. Feature Outlier Clipping (Preprocessing Tambahan)

- [x] 5.A Tambah parameter `clip_outliers: bool = False` ke `preprocess()`; jika aktif, clip setiap fitur dan y (sebelum log-transform) ke rentang [Q1 - 1.5×IQR, Q3 + 1.5×IQR] menggunakan `np.clip`
- [x] 5.B Tambah field `clip_outliers: false` ke `dataset_config.yaml` untuk setiap dataset (default `false`); set `true` hanya jika outlier signifikan terdeteksi
- [x] 5.C Teruskan flag `clip_outliers` dari config ke `preprocess()` di `train_and_export()`

## 7. Bugfix: Metric Blowup & Log-space R²

- [x] 7.1 Tambah helper `_safe_oof(pred_log, y_log)` yang clip log-space predictions ke `[y_log.min()-2, y_log.max()+2]` sebelum `expm1`; ganti semua pemanggilan `_inverse_transform(cross_val_predict(...), log_transform)` dengan ini di `evaluate_models()`
- [x] 7.2 Impor `r2_score` dari sklearn; hitung R² dari OOF predictions dalam skala asli (`r2_score(y_orig, oof_orig)`) untuk semua model; simpan ke metrics sebagai pengganti log-space R²
- [x] 7.3 Perbaiki R² di `_lstm_cv_scores_with_params`: clip before inverse-transform, hitung `r2_score(arr_true, arr_pred)` dalam skala asli
- [x] 7.4 Retrain ulang dan verifikasi LR MAE tidak lagi astronomis; verifikasi R² lebih representatif

## 6. Verifikasi & Retrain

- [x] 6.1 Jalankan `python model_pipeline.py` untuk retrain semua dataset dengan flag baru; verifikasi `_metrics.json` masing-masing mengandung `"log_transform_target"` dengan nilai yang benar
- [x] 6.2 Verifikasi metrik (MAE, MMSE) dalam `_metrics.json` berada dalam satuan asli (bukan log-space) — bandingkan dengan nilai sebelumnya: MAE seharusnya turun signifikan untuk dataset dengan skewness tinggi
- [ ] 6.3 Jalankan app, lakukan estimasi pada satu dataset — verifikasi hasil prediksi masuk akal (tidak berupa nilai log-space seperti "3.2 person-months" untuk proyek besar)
