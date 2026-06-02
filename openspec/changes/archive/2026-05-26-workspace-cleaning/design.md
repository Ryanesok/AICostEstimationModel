## Context

Root directory saat ini berisi 9 file Python dan 3 file konfigurasi YAML/txt yang tersebar rata tanpa struktur. `auto_configure.py` dan `model_pipeline.py` keduanya beroperasi pada dataset yang sama, berbagi konstanta path, dan selalu dijalankan secara berurutan (configure → train). `estimator.py` dan `downloader.py` adalah modul pendukung yang tidak berhubungan dengan UI. `app.py` satu-satunya entry point UI yang harus tetap di root agar mudah dijalankan pengguna.

## Goals / Non-Goals

**Goals:**
- Root berisi hanya `app.py` dan `README.md`
- Semua skrip non-UI tergabung dalam package `pipeline/`
- `auto_configure.py` + `model_pipeline.py` digabung menjadi `pipeline/build.py` dengan dua fungsi utama yang dapat dipanggil terpisah atau sekaligus
- File konfigurasi (`dataset_config.yaml`, `field_labels.yaml`, `datasets.txt`) dipindahkan ke `pipeline/`
- `app.py` tetap berjalan tanpa perubahan fungsional

**Non-Goals:**
- Refactor logika di dalam `auto_configure.py` atau `model_pipeline.py`
- Mengubah format atau isi file konfigurasi YAML
- Memindahkan `models/`, `data/`, atau `last_inputs/` folders
- Menambahkan fitur baru

## Decisions

### 1. `pipeline/` sebagai Python package (bukan folder biasa)
Tambahkan `pipeline/__init__.py` kosong sehingga `from pipeline.estimator import ...` bekerja tanpa manipulasi `sys.path`. Alternatif: meletakkan skrip flat di subfolder tanpa `__init__.py` dan menambahkan path secara manual — ditolak karena lebih fragile.

### 2. Gabungkan `auto_configure.py` + `model_pipeline.py` → `pipeline/build.py`
Keduanya berbagi konstanta (path dataset, path model) dan selalu dijalankan berurutan. Menggabungkan keduanya menghilangkan duplikasi dan membuat alur kerja lebih jelas. `build.py` akan mengekspos tiga entry point: `configure()`, `train()`, dan `build_all()` (keduanya sekaligus). CLI tetap bekerja via `python pipeline/build.py`.

### 3. Path konfigurasi diperbarui relatif terhadap lokasi file
Setelah dipindah ke `pipeline/`, path seperti `"dataset_config.yaml"` harus menjadi path relatif terhadap root project. Gunakan `Path(__file__).parent.parent / "pipeline" / "dataset_config.yaml"` atau konstanta `PIPELINE_DIR` di `app.py` untuk memastikan path tetap benar tanpa bergantung pada working directory.

### 4. `datasets.txt` dan `roadmap.md` dipindahkan, `test.txt` dihapus
`datasets.txt` adalah konfigurasi pipeline, cocok di `pipeline/`. `roadmap.md` dipindahkan ke `docs/` agar tidak mengotori root. `test.txt` adalah file sementara dan dihapus.

## Risks / Trade-offs

- **Import path berubah di `app.py`** → Mitigasi: perubahan minimal — hanya baris `from estimator import` dan konstanta `CONFIG_FILE`
- **Pengguna CLI perlu menyesuaikan perintah** (dari `python model_pipeline.py` ke `python pipeline/build.py`) → Mitigasi: dokumentasikan di `README.md`
- **`__pycache__`** di root menjadi stale setelah pemindahan → Mitigasi: biarkan Python meregenerasi secara otomatis; tambahkan `__pycache__/` ke `.gitignore` jika belum ada
