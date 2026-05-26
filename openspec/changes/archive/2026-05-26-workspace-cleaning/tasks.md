## 1. Buat Struktur Package pipeline/

- [x] 1.1 Buat folder `pipeline/` dan file `pipeline/__init__.py` kosong
- [x] 1.2 Salin `estimator.py` ke `pipeline/estimator.py`, pastikan semua internal path tetap valid
- [x] 1.3 Salin `downloader.py` ke `pipeline/downloader.py`, periksa referensi ke `datasets.txt`

## 2. Gabungkan auto_configure + model_pipeline → pipeline/build.py

- [x] 2.1 Buat `pipeline/build.py` dengan menyalin isi `auto_configure.py` sebagai fungsi `configure()`
- [x] 2.2 Tambahkan isi `model_pipeline.py` sebagai fungsi `train()` ke dalam `pipeline/build.py`
- [x] 2.3 Tambahkan fungsi `build_all()` yang memanggil `configure()` lalu `train()`
- [x] 2.4 Tambahkan CLI argparse di `if __name__ == "__main__"`: flag `--configure`, `--train`, default ke `build_all()`
- [x] 2.5 Perbarui semua path internal di `build.py` agar relatif terhadap root project (bukan CWD)

## 3. Pindahkan File Konfigurasi ke pipeline/

- [x] 3.1 Pindahkan `dataset_config.yaml` → `pipeline/dataset_config.yaml`
- [x] 3.2 Pindahkan `field_labels.yaml` → `pipeline/field_labels.yaml`
- [x] 3.3 Pindahkan `datasets.txt` → `pipeline/datasets.txt`
- [x] 3.4 Perbarui referensi ke `field_labels.yaml` di dalam `pipeline/build.py` (auto_configure)
- [x] 3.5 Perbarui referensi ke `datasets.txt` di dalam `pipeline/downloader.py`

## 4. Perbarui app.py

- [x] 4.1 Ubah `from estimator import ...` → `from pipeline.estimator import ...`
- [x] 4.2 Ubah konstanta `CONFIG_FILE = "dataset_config.yaml"` → `CONFIG_FILE = "pipeline/dataset_config.yaml"`
- [x] 4.3 Perbarui teks instruksi di error banner: ganti perintah CLI lama dengan path `pipeline/` baru

## 5. Bersihkan Root

- [x] 5.1 Hapus `auto_configure.py` dari root setelah `pipeline/build.py` verified
- [x] 5.2 Hapus `model_pipeline.py` dari root setelah `pipeline/build.py` verified
- [x] 5.3 Hapus `estimator.py` dari root setelah `pipeline/estimator.py` verified
- [x] 5.4 Hapus `downloader.py` dari root setelah `pipeline/downloader.py` verified
- [x] 5.5 Pindahkan `roadmap.md` → `docs/roadmap.md` (buat folder `docs/` bila perlu)
- [x] 5.6 Hapus `test.txt` dari root

## 6. Verifikasi

- [x] 6.1 Jalankan `python app.py` dan pastikan GUI terbuka tanpa error
- [x] 6.2 Jalankan `python pipeline/build.py --help` dan pastikan CLI argparse berfungsi
- [x] 6.3 Konfirmasi `python -c "from pipeline.estimator import load_estimator_models"` sukses
- [x] 6.4 Perbarui `README.md`: ganti semua referensi perintah CLI lama ke path `pipeline/` baru
