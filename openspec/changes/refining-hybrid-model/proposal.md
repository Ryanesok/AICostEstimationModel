## Why

`pipeline/build.py` sudah menghitung `hybrid_weights` (inverse-MAE dari LR + XGB + LSTM) dan menyimpannya ke metrics JSON, tetapi tidak pernah mengevaluasi performa hybrid itu sendiri secara CV — sehingga `select_best_estimator` tidak bisa memilih "Hybrid" sebagai best model karena tidak ada entri MAE/PRED25 hybrid di metrics file. Ini berarti ensemble yang sudah kita bangun tidak pernah dipakai di aplikasi.

## What Changes

- Tambahkan evaluasi CV untuk model Hybrid di `build.py`: hitung prediksi weighted menggunakan `hybrid_weights` pada setiap fold, lalu simpan `hybrid_cv_mae_mean`, `hybrid_mmre`, `hybrid_pred25` ke metrics JSON.
- Perluas `hybrid_weights` agar menyertakan SVR dan RF (saat ini hanya LR + XGB + LSTM), sehingga semua 5 model berkontribusi ke ensemble.
- Daftarkan "Hybrid" sebagai model yang dapat dipilih di `pipeline/estimator.py` (`_MODEL_MAE_KEYS`, `_MODEL_MMRE_KEYS`, `_MODEL_PRED25_KEYS`) agar `select_best_estimator` bisa mempertimbangkan Hybrid dan, jika pred25-nya ≥ 0.75, memilihnya sebagai best model.
- Pastikan `run_hybrid_estimate` di `estimator.py` sudah menangani SVR dan RF yang baru ditambahkan ke weights.

## Capabilities

### New Capabilities

- `hybrid-model-evaluation`: Evaluasi CV lengkap untuk model Hybrid (MAE, MMRE, PRED25) yang direkam di metrics JSON dan dapat dipilih oleh `select_best_estimator`.

### Modified Capabilities

- `pipeline-module`: Alur training diperluas untuk menghitung dan menyimpan metrik hybrid; model selection diperluas agar mendukung "Hybrid" sebagai kandidat.

## Impact

- `pipeline/build.py` — tambah evaluasi hybrid dalam `evaluate_models()`; perluas `hybrid_weights` ke 5 model
- `pipeline/estimator.py` — tambah "Hybrid" ke `_MODEL_MAE_KEYS`, `_MODEL_MMRE_KEYS`, `_MODEL_PRED25_KEYS`; update `run_hybrid_estimate` agar menangani SVR dan RF
- `models/*_metrics.json` — format bertambah (perlu retrain semua dataset setelah perubahan)
- Tidak ada perubahan ke UI atau `estimator_bridge.py`
