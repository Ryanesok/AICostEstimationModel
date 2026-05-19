## Context

`model_pipeline.py` sudah memiliki pola GridSearchCV yang berjalan untuk SVR dan XGBoost. Keduanya menerima objek `KFold` dari `evaluate_models()` dan mengembalikan `(best_estimator, best_params)`. RF dan LSTM saat ini tidak mengikuti pola ini:

- `train_rf(X, y)` — hardcode `n_estimators=200`, langsung fit, tidak ada grid search
- `train_lstm(X, y, input_size, ...)` — hardcode `hidden_size=64`, `lr=1e-3`; CV dilakukan terpisah di `_lstm_cv_scores()` tanpa variasi parameter

Grid search LSTM tidak bisa menggunakan `sklearn.GridSearchCV` secara langsung karena `LSTMModel` adalah PyTorch, bukan estimator scikit-learn. Opsi: (a) bungkus dengan `skorch`, (b) tulis loop manual, (c) gunakan `optuna`.

## Goals / Non-Goals

**Goals:**
- `train_rf()` menggunakan GridSearchCV atas grid kecil (≤8 kombinasi) dengan KFold yang sama dengan model lain
- `train_lstm()` melakukan grid search manual atas `hidden_size` dan `lr` (≤6 kombinasi) menggunakan CV MAE sebagai kriteria pemilihan
- `best_params` keduanya tersimpan di `_metrics.json`
- Tidak menambah dependensi baru

**Non-Goals:**
- Bayesian optimization atau random search (bukan kebutuhan saat ini)
- Tuning arsitektur LSTM yang lebih dalam (jumlah layer, dropout rate)
- Tuning Linear Regression (tidak ada hyperparameter bermakna)
- Parallel training antar dataset (tidak dalam scope ini)

## Decisions

**D1 — RF: GridSearchCV dengan grid kecil**
Grid: `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 5} → 12 kombinasi. Scoring: `neg_mean_absolute_error`, `n_jobs=-1`. Signature berubah menjadi `train_rf(X, y, kfold)` agar konsisten dengan `train_xgb`.

**D2 — LSTM: grid search manual bukan skorch/optuna**
Alasan: tidak menambah dependensi. Loop eksplisit atas `hidden_size` ∈ {32, 64, 128} dan `lr` ∈ {1e-3, 5e-4} (6 kombinasi). Untuk setiap kombinasi jalankan `_lstm_cv_scores_with_params()` yang menerima `hidden_size` dan `lr`. Pilih kombinasi dengan CV MAE minimum.

**D3 — Refaktor `_lstm_cv_scores` menjadi `_lstm_cv_scores_with_params(X, y, kfold, hidden_size, lr)`**
Fungsi saat ini tidak menerima parameter arsitektur. Perlu diperluas tanpa mengubah perilaku default (backward compatible di dalam pipeline).

**D4 — Simpan best_params di metrics.json**
Tambah key `rf_best_params: {n_estimators, max_depth, min_samples_split}` dan `lstm_best_params: {hidden_size, lr}`. Format JSON yang sudah ada tidak berubah — hanya penambahan field.

## Risks / Trade-offs

- [Waktu training naik signifikan] RF: 12 kombinasi × 5 fold × 4 dataset = 240 fit tambahan. LSTM: 6 kombinasi × 5 fold = 30 train loop tambahan. → Mitigasi: grid dibuat kecil; user diberi warning di log bahwa tuning aktif; early stopping tetap berlaku per LSTM fold.
- [Reprodusibilitas LSTM] PyTorch tidak sepenuhnya deterministik pada CPU kecuali `torch.manual_seed` diset. → Mitigasi: set seed sebelum setiap kombinasi di grid search.
- [Overfitting pada grid kecil] Grid yang sangat kecil mungkin melewatkan area optimal. → Trade-off yang diterima: prioritas adalah konsistensi proses, bukan optimal mutlak. Grid bisa diperluas di iterasi berikutnya.

## Migration Plan

Tidak ada migrasi data. Model yang sudah ada di `models/` tetap valid — hanya model baru yang ditimpa saat `model_pipeline.py` dijalankan ulang. Tidak ada perubahan pada format file `.pkl` atau `.pt`.
