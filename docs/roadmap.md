# Roadmap

Dokumen ini merangkum arah pengembangan **AI Cost Estimation Model**: apa yang sudah selesai, apa yang sedang dikerjakan, dan ke mana proyek ini akan berkembang.

Sumber kebenaran untuk perubahan terperinci ada di folder `openspec/changes/` (aktif) dan `openspec/changes/archive/` (selesai). Roadmap ini adalah ringkasan tingkat tinggi.

---

## ✅ Shipped

Fitur dan kapabilitas yang sudah selesai dan tersedia di branch utama.

### Fondasi Proyek
- Struktur proyek (CustomTkinter desktop app, pipeline training terpisah, konfigurasi YAML)
- `downloader.py` untuk mengunduh empat dataset historis (COCOMO-81, Desharnais, China, Maxwell)
- `auto_configure.py` — deteksi kolom otomatis, pemilihan target, dan pemeriksaan kebocoran fitur
- Konfigurasi dataset terpusat di `dataset_config.yaml` + label/hint di `field_labels.yaml`

### Antarmuka Aplikasi
- GUI tiga kolom (pilih dataset → input fitur → hasil estimasi & chart)
- Dynamic form: field menyesuaikan dataset yang dipilih
- Visualisasi chart untuk perbandingan model

### Model & Training
- Pipeline training dengan **5-fold cross-validation** untuk semua model
- Enam model yang didukung:
  - **SVR** — GridSearchCV (C, epsilon, kernel)
  - **Linear Regression** — OLS baseline
  - **Random Forest** — 200 estimators
  - **XGBoost** — GridSearchCV (n_estimators, max_depth, learning_rate)
  - **LSTM** — PyTorch, Adam + MSE, early stopping
  - **Hybrid** — weighted average LSTM + XGBoost + LR berdasarkan inverse CV MAE
- Pencegahan kebocoran fitur (feature leakage guard) saat training
- Metrik per model tersimpan di `models/<dataset>_metrics.json`

### Dokumentasi
- `README.md` — quick-start: prerequisites, setup, cara menjalankan, file layout, daftar model
- `docs/cocomo-81.md`, `docs/desharnais.md`, `docs/china.md`, `docs/maxwell.md` — panduan field per dataset
- Workflow OpenSpec terdokumentasi di `openspec/`

---

## 🚧 In Progress

Pekerjaan yang sedang berjalan saat ini.

- **`add-documentation`** — penambahan README, panduan dataset, dan roadmap ini (sedang difinalisasi sebelum di-archive)

---

## 🗺️ Planned

Arah pengembangan berikutnya. Daftar ini bersifat indikatif — prioritas dapat berubah berdasarkan kebutuhan riset.

### Jangka Pendek
- **Hyperparameter tuning untuk Random Forest & LSTM** — saat ini hanya SVR dan XGBoost yang di-tune via GridSearchCV
- **Export hasil estimasi** — simpan input + prediksi ke CSV/JSON dari GUI
- **Tampilan perbandingan model side-by-side** — pilih 2–3 model dan lihat prediksi serta metrik berdampingan
- **Persistensi input form** — ingat input terakhir per dataset agar tidak perlu mengetik ulang

### Jangka Menengah
- **Model explainability** — feature importance (Random Forest, XGBoost) dan SHAP values yang ditampilkan di GUI
- **Dataset tambahan** — ISBSG, NASA93, atau dataset proprietary yang dapat dikonfigurasi via `dataset_config.yaml` tanpa perubahan kode
- **Validasi input lanjutan** — peringatan ketika nilai input berada di luar rentang training data (out-of-distribution warning)
- **Localization** — opsi bahasa Inggris untuk label dan hint (saat ini hanya Indonesia)

### Jangka Panjang
- **Packaging desktop** — distribusi `.exe` / `.app` via PyInstaller agar end-user tidak perlu menginstal Python
- **Unit tests + CI** — pytest untuk pipeline training, GitHub Actions untuk lint & test
- **Versi web** — port GUI ke web (Streamlit / FastAPI + React) untuk akses tanpa instalasi
- **Active learning loop** — antarmuka untuk mencatat hasil aktual proyek dan retrain inkremental

---

## Bagaimana Berkontribusi pada Roadmap

1. Cek `openspec/changes/` untuk melihat change aktif — mungkin item yang Anda inginkan sudah dalam progress
2. Untuk ide baru, buat change baru via `/opsx:propose` — sertakan motivasi, design decision, dan task breakdown
3. Update bagian **In Progress** dan **Shipped** di roadmap ini saat change di-archive

Roadmap ini diperbarui secara manual; lihat `git log` dan `openspec/changes/archive/` untuk timeline yang akurat per-commit.
