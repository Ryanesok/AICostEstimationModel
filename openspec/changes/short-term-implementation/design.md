## Context

Pipeline training (`model_pipeline.py`) saat ini menerapkan GridSearchCV untuk SVR dan XGBoost, tetapi Random Forest menggunakan parameter tetap (`n_estimators=200`) dan LSTM menggunakan hyperparameter hardcoded. GUI (`app.py`) menampilkan hasil satu model sekaligus, tidak menyimpan state input, dan tidak ada mekanisme export. Perubahan ini bersifat additive—tidak ada breaking change pada API publik maupun format file model yang sudah ada.

## Goals / Non-Goals

**Goals:**
- RF dan LSTM menggunakan hyperparameter search yang terekam di `_metrics.json`
- Pengguna dapat mengekspor input + seluruh prediksi model ke CSV/JSON dalam satu klik
- Pengguna dapat membandingkan prediksi dan metrik beberapa model secara berdampingan
- Form terisi otomatis dari input sesi sebelumnya per dataset

**Non-Goals:**
- Perubahan pada format model `.pkl` / `.pt` yang sudah tersimpan
- Penambahan dataset baru
- Perubahan skema `dataset_config.yaml` atau `field_labels.yaml`
- Training ulang otomatis dari GUI

## Decisions

### 1. RF Tuning: `RandomizedSearchCV` bukan `GridSearchCV`

`GridSearchCV` dengan grid lebar pada RF sangat lambat (waktu training bisa 10×). `RandomizedSearchCV` dengan `n_iter=30` memberikan coverage yang cukup dalam waktu yang lebih terkontrol. Parameter grid: `n_estimators` ∈ [100, 200, 300], `max_depth` ∈ [None, 10, 20, 30], `min_samples_split` ∈ [2, 5, 10].

**Alternatif dipertimbangkan:** Optuna — terlalu berat sebagai dependensi baru untuk gain yang marginal pada dataset kecil.

### 2. LSTM Tuning: Grid search manual di atas training loop

LSTM tidak kompatibel dengan scikit-learn estimator interface. Grid search dilakukan secara manual: iterasi atas kombinasi `hidden_size` ∈ [32, 64, 128], `num_layers` ∈ [1, 2], `learning_rate` ∈ [0.001, 0.0005], pilih kombinasi dengan CV MAE terbaik.

**Alternatif dipertimbangkan:** Wrapping LSTM sebagai `sklearn.BaseEstimator` — memerlukan refaktor besar dan memperumit kode training.

### 3. Export: satu file per klik, format pilihan pengguna

Tombol export membuka `filedialog.asksaveasfilename` (sudah tersedia di tkinter). Format ditentukan dari ekstensi yang dipilih pengguna (`.csv` atau `.json`). Isi file: semua field input (label → nilai) + prediksi tiap model + timestamp.

**Alternatif dipertimbangkan:** Auto-export ke folder tetap — kurang fleksibel untuk alur kerja peneliti.

### 4. Model comparison: `CTkScrollableFrame` + Checkbutton per model

Panel hasil diperluas dengan daftar checkbox model. Setelah estimasi dijalankan, pengguna memilih model mana yang ingin dibandingkan. Tabel (`CTkLabel` grid) dan bar chart (matplotlib yang sudah dipakai) diperbarui secara reaktif. Maksimal 5 model agar layout tidak overflow.

**Alternatif dipertimbangkan:** Halaman/tab baru — menambah kompleksitas navigasi; comparison lebih intuitif di panel yang sama.

### 5. Persistence: JSON per dataset di folder `last_inputs/`

`last_inputs/<dataset_name>.json` menyimpan dict `{field_key: value}`. Dibaca saat dataset dipilih dan form diinisialisasi; ditulis setiap kali estimasi dijalankan. Folder dibuat otomatis bila belum ada.

**Alternatif dipertimbangkan:** `configparser` / SQLite — overkill untuk key-value sederhana.

## Risks / Trade-offs

- **LSTM grid search lambat** → mitigasi: batasi kombinasi (2×2×2 = 8 run), tampilkan progress log di console; training tetap dilakukan offline via `model_pipeline.py`
- **Metrics JSON sudah punya format** → best params RF/LSTM ditambahkan sebagai field baru (`best_params`) — backward compatible karena dibaca dengan `.get()`
- **`last_inputs/` bisa berisi data sensitif** → ditambahkan ke `.gitignore`; hanya nilai numerik/kategorik biasa
- **Comparison chart bisa penuh teks** jika nama model panjang → label ditruncate ke 8 karakter di sumbu X

## Migration Plan

1. Jalankan `model_pipeline.py` ulang untuk dataset yang ingin di-update agar `best_params` tersimpan di metrics JSON
2. Tidak ada perubahan pada model artifact yang sudah ada — model lama tetap bisa dipakai
3. Rollback: karena perubahan additive, mengembalikan `app.py` dan `model_pipeline.py` ke versi sebelumnya cukup via `git revert`

## Open Questions

- Apakah `last_inputs/` perlu disimpan di direktori konfigurasi sistem (`%APPDATA%`) alih-alih folder proyek? → Saat ini pakai folder proyek untuk kemudahan; bisa diubah di iterasi berikutnya.
