## Context

`pipeline/build.py` saat ini menjalankan:
1. `run_batch()` — loop `for path in csv_files: train_and_export(path)` — **sequential**
2. `evaluate_models()` — memanggil `GridSearchCV(SVR)`, `cross_validate(LR)`, `train_rf()`, `train_xgb()`, `train_lstm_with_tuning()` satu per satu — **sequential**
3. LSTM tuning — nested loop `hidden_size × num_layers × lr` (3×2×2 = 12 kombinasi), masing-masing 5-fold CV — **sequential, paling lambat**

Bottleneck utama: LSTM tuning (12 × 5-fold) + XGB GridSearch + RF RandomizedSearch berjalan satu demi satu. Dengan 3–5 dataset, total waktu ≈ jumlah dataset × waktu per dataset.

Kendala:
- PyTorch `LSTMModel` tidak aman di-`fork` lintas proses tanpa serialisasi (model state dict adalah tensor CUDA/CPU)
- sklearn estimator aman di-fork karena pure-Python/numpy
- Output artifact harus identik dengan versi sequential (deterministik via seed)
- Tidak ada dependensi baru dari luar stdlib

## Goals / Non-Goals

**Goals:**
- Dataset-level parallelism: beberapa dataset diproses bersamaan via `ProcessPoolExecutor`
- Model-level parallelism: SVR, LR, RF, XGB dijalankan bersamaan via `ThreadPoolExecutor` di dalam satu dataset
- LSTM tetap di main thread dataset-worker karena PyTorch tidak aman lintas process fork tanpa penjagaan eksplisit
- Default worker count otomatis dari `os.cpu_count()`
- CLI flags `--dataset-workers N` dan `--model-workers N` untuk tuning manual
- Logging waktu per dataset (wall-clock) di output

**Non-Goals:**
- Distributed training (multi-machine)
- GPU acceleration untuk LSTM
- Mengubah hyperparameter grid atau metrik evaluasi
- Mengubah format output artifact
- Async I/O (bukan bottleneck)

## Decisions

### D1: ProcessPoolExecutor untuk dataset-level, bukan ThreadPoolExecutor

**Pilihan:** `ProcessPoolExecutor` vs `ThreadPoolExecutor` vs `joblib.Parallel`

**Keputusan:** `ProcessPoolExecutor` untuk dataset-level.

**Alasan:** sklearn dan numpy melepas GIL di C-extension layer, tapi Python-level loops (LSTM CV) tetap di-blok GIL. Process-based parallelism memberikan isolasi memori penuh dan menghindari contention state global. `joblib.Parallel` sudah jadi dependensi (dipakai sklearn internally) tapi API `concurrent.futures` lebih explicit untuk use case ini.

**Constraint:** dataset worker function `train_and_export` harus bisa di-pickle — semua argumennya sudah berupa `str` (path), jadi aman.

---

### D2: ThreadPoolExecutor untuk model-level di dalam satu dataset

**Pilihan:** `ProcessPoolExecutor` vs `ThreadPoolExecutor` untuk SVR/LR/RF/XGB

**Keputusan:** `ThreadPoolExecutor` untuk model-level.

**Alasan:** sklearn estimators dengan `n_jobs=-1` (RF, XGB) sudah menggunakan multiple cores secara internal via C extensions — menambahkan ProcessPoolExecutor di atasnya menyebabkan over-subscription. Thread-level concurrency cukup untuk menghindari Python-level serialization overhead dan memungkinkan shared memory untuk array `X`, `y`.

LSTM tetap sekuensial di dalam dataset worker karena:
- `torch.manual_seed(42)` harus di-set tepat sebelum training untuk reproducibility
- PyTorch autograd graph tidak thread-safe jika model di-share

---

### D3: Seed strategy untuk reproducibility

**Keputusan:** Set `random_state=42` dan `torch.manual_seed(42)` di dalam masing-masing worker, bukan di main process.

**Alasan:** Tiap worker process mewarisi state dari parent saat fork, tapi urutan eksekusi bisa berbeda. Dengan seed eksplisit per worker, hasil setiap model deterministik terlepas dari urutan dataset.

---

### D4: Default worker count

**Keputusan:**
- `dataset_workers = min(4, max(1, os.cpu_count() // 2))`
- `model_workers = min(4, max(1, os.cpu_count() // 2))`

**Alasan:** Membatasi ke `cpu_count // 2` mencegah memory pressure (tiap dataset worker bisa mengonsumsi 200–500 MB RAM untuk model fitting). Cap 4 adalah batas konservatif untuk mesin 8-core standar.

---

### D5: Fallback ke sequential jika workers=1

**Keputusan:** Jika `dataset_workers=1`, `run_batch()` menggunakan loop biasa tanpa executor overhead. Sama untuk `model_workers=1`.

**Alasan:** Executor overhead (~50ms startup) tidak sepadan untuk single dataset. Mempermudah debugging karena traceback full tersedia di main thread.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Memory spike jika banyak dataset worker aktif bersamaan | Default cap di `min(4, cpu_count//2)`; expose `--dataset-workers` untuk override manual |
| Non-determinism jika seed tidak di-set per worker | Set seed eksplisit di awal setiap worker function (D3) |
| PyTorch di dalam process fork bisa deadlock di beberapa OS (khususnya macOS dengan `spawn`) | LSTM training tidak di-fork — tetap di dataset worker thread, bukan sub-process baru |
| ThreadPoolExecutor + sklearn `n_jobs=-1` bisa over-subscribe CPU | Default `model_workers=2`; user bisa set `--model-workers 1` untuk menonaktifkan |
| Error di satu dataset worker tidak menghentikan dataset lain | `ProcessPoolExecutor.map` dengan exception handling per future; gagal satu dataset → log warning, lanjut |

## Migration Plan

1. Implementasi perubahan di `pipeline/build.py` saja — tidak ada file lain yang berubah
2. Tambahkan argumen CLI `--dataset-workers` dan `--model-workers` (backward compatible, default = auto)
3. Jalankan `python pipeline/build.py --train --dataset-workers 1` untuk verifikasi bahwa output identik dengan versi lama
4. Tidak ada rollback khusus — kode lama bisa di-restore via `git revert`
