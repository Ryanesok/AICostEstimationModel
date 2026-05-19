## 1. Random Forest — GridSearchCV

- [x] 1.1 Definisikan `RF_GRID_PARAMS` di bagian konstanta `model_pipeline.py`: `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 5}
- [x] 1.2 Ubah signature `train_rf(X, y)` → `train_rf(X, y, kfold)` dan ganti body dengan `GridSearchCV(RandomForestRegressor(random_state=42), RF_GRID_PARAMS, cv=kfold, scoring="neg_mean_absolute_error", n_jobs=-1)`
- [x] 1.3 Kembalikan `(gs.best_estimator_, gs.best_params_)` dari `train_rf()` agar konsisten dengan `train_xgb()`
- [x] 1.4 Update `evaluate_models()`: teruskan `kfold` ke `train_rf()` dan tangkap `(best_rf, rf_best_params)`
- [x] 1.5 Tambah `rf_best_params` ke dict `metrics` yang dikembalikan `evaluate_models()`

## 2. LSTM — Grid Search Manual

- [x] 2.1 Ubah `LSTMModel.__init__` agar menerima `hidden_size` sebagai parameter (bukan hardcode 64)
- [x] 2.2 Ubah `train_lstm(X, y, input_size, ...)` agar menerima `hidden_size` dan `lr` sebagai parameter eksplisit (dengan default masing-masing 64 dan 1e-3 untuk backward compat)
- [x] 2.3 Buat `LSTM_GRID_PARAMS = {"hidden_size": [32, 64, 128], "lr": [1e-3, 5e-4]}` di bagian konstanta
- [x] 2.4 Buat fungsi `_lstm_cv_scores_with_params(X, y, kfold, hidden_size, lr)` — refaktor dari `_lstm_cv_scores()` dengan tambahan params; set `torch.manual_seed(42)` sebelum setiap model init
- [x] 2.5 Buat fungsi `train_lstm_with_tuning(X, y, input_size, kfold)`: loop atas semua kombinasi `LSTM_GRID_PARAMS`, panggil `_lstm_cv_scores_with_params()`, pilih kombinasi dengan CV MAE terkecil, retrain pada full data dengan params terbaik, kembalikan `(model, best_params)`
- [x] 2.6 Update `evaluate_models()`: ganti `_lstm_cv_scores(X, y, kfold)` dengan `train_lstm_with_tuning(X, y, input_size, kfold)` dan tangkap `lstm_best_params`
- [x] 2.7 Tambah `lstm_best_params` ke dict `metrics`

## 3. Persistensi & Logging

- [x] 3.1 Pastikan `rf_best_params` dan `lstm_best_params` tersimpan ke `models/<dataset>_metrics.json` (field baru, tidak menghapus field lama)
- [x] 3.2 Update log training (baris `print` di `train_one_dataset`) agar menampilkan `rf_best_params` dan `lstm_best_params` yang ditemukan

## 4. Verifikasi

- [x] 4.1 Jalankan `python model_pipeline.py` dengan satu dataset kecil (Desharnais) — pastikan tidak ada error dan `_metrics.json` memuat kedua `best_params`
- [x] 4.2 Periksa bahwa `app.py` tetap berjalan normal (load model RF dari `.pkl` tidak berubah formatnya)
- [x] 4.3 Pastikan `lstm_arch_<dataset>.json` tetap tersimpan dengan `input_size` yang benar (field `hidden_size` sekarang dinamis — perlu disimpan juga)
