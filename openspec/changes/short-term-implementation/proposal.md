## Why

Model training dan antarmuka pengguna saat ini berfungsi, tetapi beberapa gap kecil membuat pengalaman peneliti menjadi kurang efisien: dua model belum di-tuning (RF dan LSTM), hasil estimasi tidak bisa diekspor, perbandingan antar-model hanya visual satu-per-satu, dan input form hilang setiap sesi. Keempat item ini adalah prioritas jangka pendek di roadmap dan dapat diimplementasikan secara independen dalam satu change.

## What Changes

- **Random Forest**: tambah GridSearchCV untuk `n_estimators`, `max_depth`, `min_samples_split` di `model_pipeline.py`
- **LSTM**: tambah Optuna atau manual grid search untuk `hidden_size`, `num_layers`, `learning_rate`, `batch_size` di `model_pipeline.py`
- **Export estimasi**: tombol "Export" di panel hasil GUI (`app.py`) menyimpan input field + prediksi semua model ke CSV atau JSON
- **Perbandingan model side-by-side**: checkbox multi-pilih model di panel hasil; tabel + chart menampilkan prediksi dan metrik berdampingan untuk model yang dipilih
- **Persistensi input form**: input terakhir per dataset disimpan ke file JSON lokal dan di-load saat dataset dipilih kembali

## Capabilities

### New Capabilities
- `rf-lstm-hypertuning`: Hyperparameter tuning untuk Random Forest dan LSTM menggunakan grid/random search, sejajar dengan tuning SVR dan XGBoost yang sudah ada
- `estimation-export`: Export hasil estimasi (input + prediksi semua model) ke file CSV atau JSON dari GUI
- `model-comparison-view`: Tampilan side-by-side untuk memilih 2–5 model dan membandingkan prediksi serta metrik (MAE, RMSE, R²) secara berdampingan
- `form-input-persistence`: Simpan dan muat ulang input form terakhir per dataset secara otomatis antar-sesi

### Modified Capabilities

## Impact

- `model_pipeline.py` — tambah tuning loop untuk RF dan LSTM
- `app.py` — tambah tombol export, widget multi-select model, logika persistence input
- `models/<dataset>_metrics.json` — best params RF/LSTM akan ikut tersimpan
- Dependensi baru: tidak ada (Optuna opsional; bisa pakai `sklearn.model_selection.RandomizedSearchCV` yang sudah tersedia)
- File baru: `last_inputs/<dataset>.json` untuk menyimpan input terakhir per dataset
