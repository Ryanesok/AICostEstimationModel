## Why

Random Forest dan LSTM saat ini dilatih dengan hyperparameter tetap (`n_estimators=200`, `lr=1e-3`, `hidden_size=64`), sementara SVR dan XGBoost sudah menggunakan GridSearchCV. Ini menciptakan asimetri: RF dan LSTM melaporkan metrik yang mungkin suboptimal, membuat perbandingan antar model tidak setara.

## What Changes

- Tambah GridSearchCV pada `train_rf()` untuk mencari `n_estimators`, `max_depth`, dan `min_samples_split`
- Tambah grid search manual (loop over parameter combinations + CV) pada `train_lstm()` untuk `hidden_size` dan `lr`
- Simpan `best_params` RF dan LSTM ke `models/<dataset>_metrics.json` agar hasilnya dapat direproduksi
- Update pesan log training agar menampilkan parameter terbaik yang ditemukan

## Capabilities

### New Capabilities
- `rf-hyperparameter-tuning`: GridSearchCV untuk Random Forest — cari `n_estimators`, `max_depth`, `min_samples_split` menggunakan KFold yang sudah ada; kembalikan best estimator dan best_params
- `lstm-hyperparameter-tuning`: Grid search manual untuk LSTM — iterasi atas `hidden_size` dan `lr`; pilih kombinasi dengan MAE CV terbaik; kembalikan model yang dilatih ulang pada full data dengan params terbaik

### Modified Capabilities
<!-- none -->

## Impact

- `model_pipeline.py` — fungsi `train_rf()` dan `train_lstm()` berubah signature (terima `kfold` untuk RF); `evaluate_models()` perlu meneruskan `kfold` ke keduanya
- `models/<dataset>_metrics.json` — tambah field `rf_best_params` dan `lstm_best_params`
- Waktu training akan meningkat: RF grid ~6 kombinasi × 5 fold = 30 fit; LSTM grid ~4–6 kombinasi × 5 fold = 20–30 train
- Tidak ada perubahan pada `app.py`, `auto_configure.py`, atau format file model yang disimpan
