## 1. Refactor `model_pipeline.py` — Preprocessing

- [x] 1.1 `model_pipeline.py` — ubah `preprocess()` agar mengembalikan `(X, y, scaler)` saja (tanpa split); hapus `train_test_split`; scaler di-fit pada seluruh X
- [x] 1.2 `model_pipeline.py` — ganti threshold minimum row dari `< 5` menjadi `< 10`; perbarui pesan error agar menyebut "cross-validation membutuhkan minimal 10 baris"

## 2. Tambahkan Cross-Validation & GridSearchCV

- [x] 2.1 `model_pipeline.py` — buat fungsi `evaluate_models(X, y, scaler)` yang menjalankan `KFold(n_splits=5, shuffle=True, random_state=42)` dan mengembalikan dict metrik CV untuk SVR dan LinearRegression (`cv_mae_mean`, `cv_mae_std`, `cv_r2_mean`, `cv_r2_std`)
- [x] 2.2 `model_pipeline.py` — dalam `evaluate_models()`, gunakan `GridSearchCV` untuk SVR dengan grid `C=[0.1, 1, 10, 100]`, `epsilon=[0.01, 0.1, 1.0]`, `kernel=['rbf', 'linear']`; scoring `neg_mean_absolute_error`; simpan `best_params_` di dict hasil
- [x] 2.3 `model_pipeline.py` — hapus fungsi `train_models()` dan `evaluate()` lama; gantikan dengan pemanggilan `evaluate_models()` di `train_and_export()`

## 3. Final Model — Train on Full Data

- [x] 3.1 `model_pipeline.py` — setelah `evaluate_models()` selesai, fit ulang SVR terbaik (dengan `best_params_`) dan LinearRegression baru pada **seluruh** X (bukan fold); simpan ke pkl seperti biasa
- [x] 3.2 `model_pipeline.py` — tambahkan konstanta `GRID_PARAMS` di bagian atas file berisi dict parameter grid SVR agar mudah diubah tanpa masuk ke fungsi

## 4. Ekspor Metrik JSON

- [x] 4.1 `model_pipeline.py` — buat fungsi `save_metrics(stem, metrics_dict)` yang menulis `models/<stem>_metrics.json` berisi: `cv_mae_mean`, `cv_mae_std`, `cv_r2_mean`, `cv_r2_std`, `best_svr_params`, `lr_cv_mae_mean`, `lr_cv_r2_mean`, `trained_on_rows`
- [x] 4.2 `model_pipeline.py` — panggil `save_metrics()` di `train_and_export()` setelah pkl berhasil ditulis
- [x] 4.3 `model_pipeline.py` — perbarui output `evaluate()` / print di `train_and_export()` agar mencetak metrik CV dan best SVR params ke stdout dalam format terbaca

## 5. Feature Leakage Guard di `auto_configure.py`

- [x] 5.1 `auto_configure.py` — tambahkan fungsi `check_leakage(df, features, target, threshold=0.95)` yang menghitung `abs(pearsonr(feature, target))` untuk setiap fitur; mengembalikan list fitur dengan |r| ≥ threshold beserta nilai r-nya
- [x] 5.2 `auto_configure.py` — dalam `configure_one()`, setelah target dipilih, panggil `check_leakage()` dan untuk setiap fitur terindikasi: cetak warning dengan nilai korelasi, minta konfirmasi user untuk keep/drop; hapus fitur yang di-drop dari list `features`
- [x] 5.3 `auto_configure.py` — jika setelah leakage check tidak ada fitur tersisa, cetak pesan error dan `return None`

## 6. Near-Constant Column Filter di `auto_configure.py`

- [x] 6.1 `auto_configure.py` — tambahkan fungsi `find_near_constant(df, candidates, threshold=0.95)` yang mengembalikan list kolom dimana nilai paling sering muncul ≥ 95% dari total non-null baris
- [x] 6.2 `auto_configure.py` — dalam `configure_one()`, sebelum menampilkan daftar fitur ke user, panggil `find_near_constant()` dan keluarkan hasilnya dari `candidates`; cetak notice untuk setiap kolom yang dihapus

## 7. Cleanup `cleanup_stale_models()`

- [x] 7.1 `model_pipeline.py` — perbarui `cleanup_stale_models()` agar juga menghapus `<stem>_metrics.json` jika ada, selain trio pkl

## 8. Default Labels/Hints File

- [x] 8.1 Buat file `field_labels.yaml` di root project berisi labels, hints, dan effort_unit untuk semua 4 dataset (cocomo-81, desharnais, china, maxwell)
- [x] 8.2 `auto_configure.py` — tambahkan konstanta `FIELD_LABELS_FILE = "field_labels.yaml"` dan fungsi `load_default_labels(stem=None)` yang membaca file tersebut
- [x] 8.3 `auto_configure.py` — update `collect_labels_hints(features, stem="")` agar: jika user melewati (jawab 'N'), kembalikan defaults dari `field_labels.yaml`; jika user mengisi manual, tampilkan nilai default sebagai placeholder dalam prompt
- [x] 8.4 `auto_configure.py` — update pemanggilan `collect_labels_hints()` di `configure_one()` agar meneruskan `stem`
- [x] 8.5 `auto_configure.py` — tambahkan fungsi `backfill_defaults(existing)` yang mengisi labels, hints, dan effort_unit dari `field_labels.yaml` untuk dataset yang sudah ada dalam config tapi belum punya nilai tersebut; panggil di `main()` sebelum save

## 9. Primary/Secondary Target

- [x] 9.1 `field_labels.yaml` — tambahkan key `secondary_target` untuk china (Duration) dan maxwell (Duration); tambahkan juga `secondary_target_candidates` untuk maxwell ([Duration, Time]) agar leakage lebih lengkap
- [x] 9.2 `auto_configure.py` — update `backfill_defaults()`: jika defaults punya `secondary_target`, tambahkan ke config DAN hapus dari `features` jika masih ada di sana; cetak notice per kolom yang dipindahkan
- [x] 9.3 `auto_configure.py` — di `configure_one()`, setelah primary target dipilih dan leakage check selesai, tawarkan pemilihan secondary target opsional; secondary target dihapus dari `features`; simpan ke config sebagai `secondary_target`
- [x] 9.4 `model_pipeline.py` — sebelum memanggil `preprocess()`, cek eksplisit `config["target"] in df.columns`; jika tidak ada, cetak pesan jelas dan `return False`
- [x] 9.5 `model_pipeline.py` — di `train_and_export()`, sebelum `preprocess()`, keluarkan `secondary_target` dari features list jika masih ada (guard runtime)
- [x] 9.6 Jalankan `python auto_configure.py` — konfirmasi backfill secondary_target berhasil; Duration hilang dari features china dan maxwell

## 10. Split Target Keyword Sets

- [x] 10.1 `auto_configure.py` — pecah `EFFORT_COLS` menjadi `PRIMARY_TARGET_COLS` (effort, cost, man-months, dll.) dan `SECONDARY_TARGET_COLS` (duration, time, elapsed); hapus `EFFORT_COLS`
- [x] 10.2 `auto_configure.py` — update `suggest_target()`: loop `PRIMARY_TARGET_COLS` dulu; jika tidak ada match, loop `SECONDARY_TARGET_COLS`; jika masih tidak ada, fallback ke korelasi
- [x] 10.3 `auto_configure.py` — update `is_effort_col()` → `is_primary_target_col()` dan tambahkan `is_secondary_target_col()`; perbarui semua pemanggilan; update `suggested_reason` agar membedakan "primary keyword" vs "secondary keyword" vs "korelasi"

## 11. Verifikasi

- [x] 9.1 Jalankan `python model_pipeline.py` — konfirmasi semua 4 dataset berhasil dilatih dan file `models/<stem>_metrics.json` terbuat
- [x] 9.2 Periksa isi salah satu `_metrics.json` — konfirmasi berisi `cv_mae_mean`, `best_svr_params`, `trained_on_rows`
- [x] 11.1 Jalankan `python model_pipeline.py` — konfirmasi semua 4 dataset berhasil dilatih dan file `models/<stem>_metrics.json` terbuat
- [x] 11.2 Periksa isi salah satu `_metrics.json` — konfirmasi berisi `cv_mae_mean`, `best_svr_params`, `trained_on_rows`
- [x] 11.3 Jalankan `python auto_configure.py` — konfirmasi backfill labels/hints/effort_unit diterapkan ke semua 4 dataset yang ada di config
- [ ] 11.4 Jalankan `python auto_configure.py` pada dataset baru dengan kolom mencurigakan — konfirmasi warning leakage muncul dan defaults digunakan jika user melewati labels
- [ ] 11.5 Jalankan `python app.py` — konfirmasi aplikasi berjalan normal dan label kolom tampil dengan benar
