## Why

Seiring bertambahnya dataset dan model, root directory sudah dipenuhi banyak skrip pipeline yang tidak terkait langsung dengan UI — membuat navigasi dan onboarding menjadi membingungkan. Workspace perlu dirapikan sekarang sebelum lebih banyak skrip ditambahkan dan ketergantungan antar-file semakin kompleks.

## What Changes

- Buat folder `pipeline/` untuk menampung semua skrip non-UI yang berhubungan dengan data dan model
- Gabungkan `auto_configure.py` dan `model_pipeline.py` menjadi satu modul (`pipeline/build.py`) karena keduanya selalu dijalankan berurutan dan berbagi logika konfigurasi
- Pindahkan `downloader.py` → `pipeline/downloader.py`
- Pindahkan `estimator.py` → `pipeline/estimator.py`
- Pindahkan file konfigurasi data (`dataset_config.yaml`, `field_labels.yaml`, `datasets.txt`) → `pipeline/`
- Sisakan hanya `app.py` dan `README.md` di root
- Perbarui semua import di `app.py` agar mengarah ke `pipeline/`
- Hapus file sementara yang tidak relevan (`test.txt`, `roadmap.md`) dari root

## Capabilities

### New Capabilities
- `pipeline-module`: Modul `pipeline/` sebagai satu paket Python yang mengekspos semua fungsi estimasi, konfigurasi, dan download dataset

### Modified Capabilities
<!-- Tidak ada perubahan requirements level-spec; ini murni reorganisasi struktur file -->

## Impact

- `app.py` perlu memperbarui import: `from estimator import ...` → `from pipeline.estimator import ...`, dan path ke config files
- Pengguna yang menjalankan skrip CLI perlu menggunakan path baru, e.g. `python pipeline/build.py`
- `requirements.txt` tidak berubah
- Folder `models/`, `data/`, `last_inputs/` tidak dipindahkan
