## 1. Kembalikan OOF Predictions dari Setiap _eval_* di build.py

Setiap fungsi `_eval_*` sudah menghitung OOF predictions dalam skala original (`svr_oof`, `lr_oof`, `rf_oof`, `xgb_oof`) tetapi tidak mengembalikannya. Tambahkan field `"<model>_oof"` ke dict hasil masing-masing.

- [x] 1.1 `pipeline/build.py` — di `_eval_svr()`, tambahkan `"svr_oof": svr_oof` ke dict yang dikembalikan (svr_oof sudah dihitung di baris yang ada, hanya perlu disertakan dalam return dict)
- [x] 1.2 `pipeline/build.py` — di `_eval_lr()`, tambahkan `"lr_oof": lr_oof` ke dict yang dikembalikan
- [x] 1.3 `pipeline/build.py` — di `_eval_rf()`, tambahkan `"rf_oof": rf_oof` ke dict yang dikembalikan
- [x] 1.4 `pipeline/build.py` — di `_eval_xgb()`, tambahkan `"xgb_oof": xgb_oof` ke dict yang dikembalikan
- [x] 1.5 `pipeline/build.py` — pastikan LSTM OOF tersedia: di `evaluate_models()`, panggil `cross_val_predict` dengan model LSTM terbaik setelah `train_lstm_with_tuning` untuk mendapatkan lstm_oof dalam skala original; simpan ke variabel `lstm_oof` (numpy array)

## 2. Hitung Hybrid Weights (5 Model) dan Metrik CV Hybrid di build.py

Setelah semua OOF tersedia, hitung bobot inverse-MAE untuk 5 model dan evaluasi performa hybrid secara CV.

- [x] 2.1 `pipeline/build.py` — di `evaluate_models()`, setelah semua `_eval_*` selesai, hitung `w_svr = 1 / max(svr_mae, 1e-6)`, `w_lr`, `w_rf`, `w_xgb`, `w_lstm` lalu normalisasi total ke 1.0; ganti `hybrid_weights` yang lama (hanya lr/xgb/lstm) dengan dict baru `{"svr": w_svr/total, "lr": ..., "rf": ..., "xgb": ..., "lstm": ...}`
- [x] 2.2 `pipeline/build.py` — di `evaluate_models()`, hitung `hybrid_oof` sebagai weighted average dari `svr_oof * w_svr + lr_oof * w_lr + rf_oof * w_rf + xgb_oof * w_xgb + lstm_oof * w_lstm` (semua sudah dalam skala original); hitung `hybrid_mae = float(mean_absolute_error(y_orig, hybrid_oof))`, `hybrid_mmre, hybrid_pred25 = _mmre_pred25(y_orig, hybrid_oof)`
- [x] 2.3 `pipeline/build.py` — tambahkan `"hybrid_cv_mae_mean": hybrid_mae`, `"hybrid_mmre": hybrid_mmre`, `"hybrid_pred25": hybrid_pred25` ke dict yang dikembalikan `evaluate_models()`

## 3. Update Print Output di build.py

- [x] 3.1 `pipeline/build.py` — di `train_and_export()`, perbarui baris print hybrid weights agar menampilkan semua 5 model: `f"  [{stem}] Hybrid CV-MAE={metrics['hybrid_cv_mae_mean']:.2f}  PRED25={metrics['hybrid_pred25']:.4f}  weights=svr={hw['svr']:.3f} lr={hw['lr']:.3f} rf={hw['rf']:.3f} xgb={hw['xgb']:.3f} lstm={hw['lstm']:.3f}"`

## 4. Daftarkan Hybrid ke Model Selection di estimator.py

- [x] 4.1 `pipeline/estimator.py` — tambahkan entry `"Hybrid": "hybrid_cv_mae_mean"` ke `_MODEL_MAE_KEYS`
- [x] 4.2 `pipeline/estimator.py` — tambahkan entry `"Hybrid": "hybrid_mmre"` ke `_MODEL_MMRE_KEYS`
- [x] 4.3 `pipeline/estimator.py` — tambahkan entry `"Hybrid": "hybrid_pred25"` ke `_MODEL_PRED25_KEYS`
- [x] 4.4 `pipeline/estimator.py` — tambahkan entry `"Hybrid": None` ke `_MODEL_HYPERPARAMS_KEYS` (Hybrid tidak punya hyperparams tunggal)

## 5. Update run_hybrid_estimate agar Mendukung SVR dan RF

- [x] 5.1 `pipeline/estimator.py` — di `run_hybrid_estimate()`, tambahkan blok untuk SVR: `if models.svr is not None and weights.get("svr", 0) > 0: v = float(models.svr.predict(X_scaled)[0]); raw["svr"] = math.expm1(v) if models.log_transform_target else v`
- [x] 5.2 `pipeline/estimator.py` — di `run_hybrid_estimate()`, tambahkan blok serupa untuk RF: `if models.rf is not None and weights.get("rf", 0) > 0: v = float(models.rf.predict(X_scaled)[0]); raw["rf"] = ...`

## 6. Verifikasi

- [x] 6.1 Retrain salah satu dataset kecil (misalnya kemerer) dan periksa `models/kemerer_metrics.json` — konfirmasi ada field `hybrid_cv_mae_mean`, `hybrid_pred25`, dan `hybrid_weights` memiliki 5 key (svr/lr/rf/xgb/lstm)
- [x] 6.2 Jalankan `python -c "from pipeline.estimator import select_best_estimator; from core.estimator_bridge import SUPPORTED_STEMS; r = select_best_estimator('models', supported_stems=SUPPORTED_STEMS); print(r.stem, r.model_name, r.pred25)"` — konfirmasi Hybrid muncul sebagai kandidat dan dipilih jika pred25-nya tertinggi
- [x] 6.3 Retrain semua dataset: `python pipeline/build.py --train` — konfirmasi tidak ada error dan semua metrics JSON terupdate

## 7. Tambahkan KNN sebagai Model Individual di build.py

Dataset berukuran kecil cocok dengan KNN (non-parametrik, berbasis kemiripan). KNN ditambahkan sebagai model yang dievaluasi dan diekspor, serta menggantikan LR/XGB/LSTM dalam komposisi Hybrid.

- [x] 7.1 `pipeline/build.py` — tambahkan `from sklearn.neighbors import KNeighborsRegressor` ke blok import
- [x] 7.2 `pipeline/build.py` — tambahkan konstanta `KNN_GRID_PARAMS = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}` di seksi grid params
- [x] 7.3 `pipeline/build.py` — buat fungsi `_eval_knn(X, y, kfold, log_transform)` mengikuti pola `_eval_svr`: jalankan `GridSearchCV(KNeighborsRegressor(), KNN_GRID_PARAMS, ...)`, hitung `knn_oof` via `cross_val_predict`, hitung `knn_mae/mmre/pred25`, kembalikan dict dengan `knn_cv_mae_mean`, `knn_cv_r2_mean`, `knn_best_params`, `knn_mmre`, `knn_pred25`, `knn_oof`
- [x] 7.4 `pipeline/build.py` — di `evaluate_models()`, tambahkan `knn_r = _eval_knn(X, y, kfold, log_transform)` di kedua path (parallel dan sequential); sertakan `**{k: v for k, v in knn_r.items() if k != "knn_oof"}` ke return dict
- [x] 7.5 `pipeline/build.py` — di `train_and_export()`, latih model KNN final: `final_knn = KNeighborsRegressor(**metrics["knn_best_params"]); final_knn.fit(X, y)` lalu ekspor dengan `joblib.dump(final_knn, os.path.join(MODELS_DIR, f"knn_{stem}.pkl"))`
- [x] 7.6 `pipeline/build.py` — di `train_and_export()`, tambahkan baris print KNN: `f"  [{stem}] KNN  CV-MAE={metrics['knn_cv_mae_mean']:.2f}  CV-R²={metrics['knn_cv_r2_mean']:.4f}  params={metrics['knn_best_params']}"`
- [x] 7.7 `pipeline/build.py` — di `cleanup_stale_models()`, tambahkan `"knn_"` ke loop prefix agar file `knn_<stem>.pkl` turut dihapus saat dataset dihapus

## 8. Ganti Hybrid Composition ke SVR + RF + KNN di build.py dan estimator.py

Hybrid baru hanya menggunakan 3 model yang paling cocok untuk dataset kecil: SVR (margin-based), RF (bagging nonlinear), KNN (instance-based).

- [x] 8.1 `pipeline/build.py` — di `evaluate_models()`, ganti blok hybrid weights: hapus kontribusi `lr`, `xgb`, `lstm`; hitung hanya `w_svr`, `w_rf`, `w_knn` dengan inverse-MAE dan normalisasi; simpan ke `hybrid_weights = {"svr": ..., "rf": ..., "knn": ...}`
- [x] 8.2 `pipeline/build.py` — di `evaluate_models()`, perbarui `hybrid_oof` menjadi weighted average dari `svr_oof`, `rf_oof`, `knn_oof` saja
- [x] 8.3 `pipeline/build.py` — di `train_and_export()`, perbarui baris print hybrid agar menampilkan: `f"  [{stem}] Hybrid CV-MAE={metrics['hybrid_cv_mae_mean']:.2f}  PRED25={metrics['hybrid_pred25']:.4f}  weights=svr={hw['svr']:.3f} rf={hw['rf']:.3f} knn={hw['knn']:.3f}"`
- [x] 8.4 `pipeline/estimator.py` — tambahkan field `knn: object = None` ke dataclass `LoadedModels`
- [x] 8.5 `pipeline/estimator.py` — tambahkan `"KNN": "knn_cv_mae_mean"` ke `_MODEL_MAE_KEYS`; `"KNN": "knn_mmre"` ke `_MODEL_MMRE_KEYS`; `"KNN": "knn_pred25"` ke `_MODEL_PRED25_KEYS`; `"KNN": "knn_best_params"` ke `_MODEL_HYPERPARAMS_KEYS`
- [x] 8.6 `pipeline/estimator.py` — di `load_estimator_models()`, muat `knn_{stem}.pkl` jika ada: `knn_path = ...; if os.path.exists(knn_path): m.knn = joblib.load(knn_path)`
- [x] 8.7 `pipeline/estimator.py` — di `run_estimate()`, tambahkan blok `if model_name == "KNN": return _postprocess(float(models.knn.predict(X_scaled)[0]))`
- [x] 8.8 `pipeline/estimator.py` — di `run_hybrid_estimate()`, tambahkan blok KNN: `if models.knn is not None and weights.get("knn", 0) > 0: v = float(models.knn.predict(X_scaled)[0]); raw["knn"] = math.expm1(v) if models.log_transform_target else v`

## 9. Verifikasi

- [x] 9.1 Retrain kemerer dan periksa `models/kemerer_metrics.json` — konfirmasi ada `knn_cv_mae_mean`, `hybrid_weights` hanya punya key `svr/rf/knn`
- [x] 9.2 Jalankan `select_best_estimator` — konfirmasi KNN muncul sebagai kandidat individual dan Hybrid muncul dengan komposisi baru
- [x] 9.3 Retrain semua dataset: `python pipeline/build.py --train`

## 10. Thresholding Bobot Hybrid

Model yang MAE-nya jauh lebih buruk dari model terbaik akan diberi bobot 0 agar tidak melemahkan Hybrid. Semua 4 model (SVR, RF, KNN, XGB) kini menjadi kandidat pool Hybrid — thresholding yang memilih siapa yang aktif.

- [x] 10.1 `pipeline/build.py` — tambahkan konstanta `HYBRID_MAE_THRESHOLD = 0.30` di seksi grid params (model dengan MAE > `best_mae × (1 + threshold)` akan dikecualikan dari Hybrid)
- [x] 10.2 `pipeline/build.py` — di `evaluate_models()`, ganti blok hybrid weights menjadi: hitung `maes = {"svr": ..., "rf": ..., "knn": ..., "xgb": ...}` dan `oofs = {..}` untuk 4 model; temukan `best_mae = min(maes.values())`; filter `active = {k: mae for k, mae in maes.items() if mae <= best_mae * (1 + HYBRID_MAE_THRESHOLD)}`; jika `active` kosong fallback ke model terbaik saja
- [x] 10.3 `pipeline/build.py` — hitung inverse-MAE weights hanya untuk model aktif, normalisasi; simpan ke `hybrid_weights = {k: norm_weights.get(k, 0.0) for k in maes}` (4 key selalu ada, inactive = 0.0)
- [x] 10.4 `pipeline/build.py` — hitung `hybrid_oof = sum(norm_weights[k] * oofs[k] for k in active)`; lanjutkan hitung hybrid_mae, hybrid_mmre, hybrid_pred25 seperti sebelumnya

## 11. Perluas Tuning SVR dan KNN

- [x] 11.1 `pipeline/build.py` — ubah `GRID_PARAMS` (SVR) menjadi list of dicts untuk mendukung `gamma` hanya pada kernel rbf: `[{"C": [0.1, 1, 10, 100, 1000], "epsilon": [0.01, 0.1, 1.0], "kernel": ["rbf"], "gamma": ["scale", "auto"]}, {"C": [0.1, 1, 10, 100, 1000], "epsilon": [0.01, 0.1, 1.0], "kernel": ["linear"]}]`
- [x] 11.2 `pipeline/build.py` — perluas `KNN_GRID_PARAMS`: `{"n_neighbors": [2, 3, 5, 7, 10, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}`

## 12. Verifikasi

- [x] 12.1 Retrain kitchenham dan periksa `models/kitchenham_metrics.json` — konfirmasi `hybrid_weights` hanya memiliki model kompetitif (weight > 0), Hybrid MAE tidak lebih buruk dari model terbaik individual
- [x] 12.2 Retrain semua dataset: `python pipeline/build.py --train`
