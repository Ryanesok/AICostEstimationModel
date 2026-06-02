## Context

Training pipeline (`pipeline/build.py`) sudah menghitung `hybrid_weights` berbasis inverse-MAE untuk LR, XGB, dan LSTM, dan menyimpannya ke `models/*_metrics.json`. Namun:

1. Tidak ada CV evaluation untuk model Hybrid itu sendiri — sehingga `hybrid_cv_mae_mean`, `hybrid_mmre`, `hybrid_pred25` tidak ada di metrics JSON.
2. `_MODEL_MAE_KEYS` di `estimator.py` tidak mendaftarkan "Hybrid", sehingga `select_best_estimator` tidak pernah mempertimbangkan Hybrid sebagai kandidat.
3. SVR dan RF dikecualikan dari `hybrid_weights` meskipun keduanya sudah dilatih dan disimpan.
4. `run_hybrid_estimate` hanya membaca bobot `lr`, `xgb`, `lstm` — tidak siap untuk SVR/RF.

## Goals / Non-Goals

**Goals:**
- Evaluasi performa Hybrid (MAE, MMRE, PRED25) secara CV di training pipeline
- Perluas `hybrid_weights` ke 5 model (SVR, LR, RF, XGB, LSTM)
- Daftarkan "Hybrid" sebagai kandidat selectable di `select_best_estimator`
- Update `run_hybrid_estimate` agar bisa membaca bobot SVR dan RF

**Non-Goals:**
- Stacking (meta-learner di atas prediksi fold) — terlalu kompleks untuk dataset SEE kecil
- Perubahan ke UI atau `estimator_bridge.py`
- Optimasi bobot via grid-search — inverse-MAE sudah cukup deterministik dan efisien

## Decisions

### 1. Inverse-MAE sebagai bobot (bukan optimasi)

Bobot dihitung sebagai `w_i = 1 / max(cv_mae_i, 1e-6)` lalu dinormalisasi. Alternatif: scipy minimize atau Optuna untuk mencari bobot optimal, tapi ini membutuhkan fold terpisah untuk validasi bobot dan rentan overfit pada dataset kecil. Inverse-MAE lebih sederhana, deterministik, dan cukup baik untuk dataset SEE berukuran < 200 baris.

### 2. Evaluasi hybrid via fold predictions yang sudah ada

Setiap model sudah punya `cross_val_predict` atau fold-level predictions dalam fungsi `_eval_*`. Pendekatan paling bersih: kumpulkan prediksi fold dari setiap model dalam dict, lalu hitung prediksi hybrid per-sample sebagai weighted average sebelum inverse-transform, kemudian hitung MAE, MMRE, PRED25 dari prediksi hybrid tersebut vs. y aktual.

Implementasi konkret: tambahkan parameter `return_oof` (out-of-fold predictions) ke setiap fungsi `_eval_*`, kumpulkan di `evaluate_models`, hitung hybrid OOF predictions menggunakan weights, lalu hitung metrik.

### 3. Tambahkan SVR dan RF ke hybrid_weights

SVR memiliki performa baik pada dataset kecil; RF baik pada data nonlinear. Kedua model sudah dilatih dan disimpan. Menambahkannya ke ensemble memperluas keragaman model tanpa biaya training tambahan.

### 4. Tambahkan "Hybrid" ke _MODEL_MAE_KEYS

Kunci baru di dict: `"Hybrid": "hybrid_cv_mae_mean"`. Demikian pula untuk MMRE dan PRED25. Ini cukup untuk membuat `select_best_estimator` mempertimbangkan Hybrid tanpa perubahan logika seleksi.

## Risks / Trade-offs

- [Risiko] OOF predictions perlu dikumpulkan dari setiap `_eval_*` yang terpisah → memerlukan refactor ringan pada interface setiap fungsi evaluasi.  
  **Mitigasi**: Kembalikan OOF sebagai field opsional dalam dict hasil; jika tidak ada (misal karena exception), hybrid evaluation dilewati dan `hybrid_cv_mae_mean` tidak disimpan.

- [Trade-off] Hybrid dengan 5 model sedikit lebih lambat saat inference (`run_hybrid_estimate` harus menjalankan 5 model) vs. single model.  
  **Dampak**: Negligible — semua model sudah dimuat di memory, prediksi untuk satu sampel sangat cepat.

- [Trade-off] Menambah SVR ke hybrid bisa mengecilkan kontribusi LSTM/XGB jika SVR dominan di dataset tertentu.  
  **Mitigasi**: Bobot dinormalisasi otomatis; jika SVR MAE buruk, bobotnya kecil.

- [Risiko] Semua model dalam `models/` harus dilatih ulang setelah perubahan agar metrics JSON punya field hybrid baru.  
  **Mitigasi**: Dokumentasikan bahwa retrain diperlukan; `select_best_estimator` gracefully melewati model tanpa `hybrid_cv_mae_mean` (key tidak ada = None, tidak dimasukkan kandidat).

## Migration Plan

1. Implementasikan perubahan di `build.py` dan `estimator.py`
2. Retrain semua dataset: `python pipeline/build.py --train`
3. Verifikasi `models/*_metrics.json` punya field `hybrid_cv_mae_mean`, `hybrid_pred25`
4. Verifikasi `select_best_estimator` memilih Hybrid jika pred25-nya lebih baik
