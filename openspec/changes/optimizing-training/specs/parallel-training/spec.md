## ADDED Requirements

### Requirement: Parallel dataset processing
Sistem SHALL memproses beberapa dataset secara bersamaan menggunakan `ProcessPoolExecutor` ketika `dataset_workers > 1`. Setiap dataset worker berjalan di proses terpisah sehingga crash atau error satu dataset tidak menghentikan dataset lain.

#### Scenario: Multiple datasets diproses paralel
- **WHEN** terdapat lebih dari satu file CSV di `data/` dan `dataset_workers > 1`
- **THEN** beberapa dataset mulai diproses bersamaan dan total waktu wall-clock lebih kecil dari jumlah waktu per dataset

#### Scenario: Error satu dataset tidak menghentikan proses
- **WHEN** salah satu dataset gagal (mis. kolom tidak ditemukan) dan dataset lain valid
- **THEN** sistem mencatat warning untuk dataset yang gagal dan melanjutkan training dataset lain hingga selesai

#### Scenario: Fallback ke sequential jika dataset_workers=1
- **WHEN** `--dataset-workers 1` diset atau hanya ada satu dataset
- **THEN** sistem berjalan tanpa `ProcessPoolExecutor` overhead dan output identik dengan versi non-parallel

### Requirement: Parallel model evaluation
Di dalam satu dataset, sistem SHALL mengevaluasi SVR, LR, RF, dan XGB secara bersamaan menggunakan `ThreadPoolExecutor` ketika `model_workers > 1`. LSTM dijalankan sekuensial di thread dataset yang sama.

#### Scenario: Model-level parallelism aktif
- **WHEN** `model_workers > 1` dan dataset memiliki cukup data
- **THEN** SVR GridSearch, LR cross-validate, RF RandomizedSearch, dan XGB GridSearch dijalankan bersamaan dalam thread pool

#### Scenario: LSTM tidak ikut thread pool
- **WHEN** model evaluation berjalan dengan `model_workers > 1`
- **THEN** LSTM training dan tuning tetap berjalan di thread dataset worker utama (bukan di thread pool model)

#### Scenario: Fallback ke sequential jika model_workers=1
- **WHEN** `--model-workers 1` diset
- **THEN** model dievaluasi satu per satu seperti sebelumnya, hasil identik

### Requirement: Configurable worker counts via CLI
Sistem SHALL menyediakan argumen CLI `--dataset-workers` dan `--model-workers` untuk mengontrol jumlah worker. Jika tidak diset, default dihitung otomatis sebagai `min(4, max(1, os.cpu_count() // 2))`.

#### Scenario: Default worker count digunakan
- **WHEN** pengguna menjalankan `python pipeline/build.py --train` tanpa flag worker
- **THEN** sistem menentukan jumlah worker secara otomatis berdasarkan `os.cpu_count()`

#### Scenario: Manual worker count diset
- **WHEN** pengguna menjalankan `python pipeline/build.py --train --dataset-workers 2 --model-workers 2`
- **THEN** sistem menggunakan tepat 2 dataset workers dan 2 model workers

### Requirement: Training time logging
Sistem SHALL mencetak waktu wall-clock yang dihabiskan per dataset dan total waktu training di akhir `run_batch()`.

#### Scenario: Waktu per dataset dicetak
- **WHEN** training selesai untuk satu dataset
- **THEN** sistem mencetak baris seperti `[stem] Selesai dalam X.XX detik`

#### Scenario: Total waktu dicetak di akhir
- **WHEN** semua dataset selesai diproses
- **THEN** sistem mencetak total waktu wall-clock, jumlah dataset berhasil, dan jumlah dataset gagal

### Requirement: Reproducibility dengan parallel execution
Sistem SHALL menghasilkan model dan metrik yang deterministik antara eksekusi sequential dan parallel. Setiap worker SHALL men-set `random_state=42` dan `torch.manual_seed(42)` secara independen di awal eksekusi.

#### Scenario: Hasil parallel identik dengan sequential
- **WHEN** training dijalankan dengan `--dataset-workers 1 --model-workers 1`
- **THEN** file `.pkl`, `.pt`, dan `_metrics.json` yang dihasilkan identik bit-per-bit dengan hasil tanpa flag parallel
