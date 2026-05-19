## 1. README.md (Root)

- [x] 1.1 Buat `README.md` di root — bagian: deskripsi proyek dan tujuan
- [x] 1.2 `README.md` — bagian: Prerequisites (Python 3.11+, pip, daftar dependensi utama)
- [x] 1.3 `README.md` — bagian: Setup & Cara Menjalankan (4 langkah: downloader → auto_configure → model_pipeline → app)
- [x] 1.4 `README.md` — bagian: File Layout (tree sederhana yang mencakup `app.py`, `model_pipeline.py`, `auto_configure.py`, `data/`, `models/`, `docs/`, `field_labels.yaml`, `dataset_config.yaml`)
- [x] 1.5 `README.md` — bagian: Model yang Tersedia (tabel 6 model: SVR, LR, RF, XGBoost, LSTM, Hybrid — singkat per baris)

## 2. docs/ — Panduan Dataset

- [x] 2.1 Buat folder `docs/` di root proyek
- [x] 2.2 Buat `docs/cocomo-81.md` — overview dataset, effort_unit person-months, tabel 16 field (Field | Label | Unit/Skala | Nilai Diterima | Contoh)
- [x] 2.3 Buat `docs/desharnais.md` — overview dataset, effort_unit person-hours, catatan konversi ÷160, tabel 5 field
- [x] 2.4 Buat `docs/china.md` — overview dataset, effort_unit person-hours, catatan konversi ÷160, secondary_target Duration, tabel 14 field input (tanpa Duration)
- [x] 2.5 Buat `docs/maxwell.md` — overview dataset, effort_unit person-hours, catatan konversi ÷160, secondary_target Duration, tabel 25 field input (tanpa Duration), penjelasan singkat skala 1–5 untuk T-factor

## 3. Verifikasi

- [x] 3.1 Konfirmasi `README.md` ada di root dan keempat file `docs/*.md` tersedia
- [x] 3.2 Periksa tabel di setiap panduan — semua fitur dari `dataset_config.yaml` tercakup tanpa ada field yang terlewat
