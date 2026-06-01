## 1. CLI Arguments

- [x] 1.1 Tambahkan argumen `--dataset-workers` ke `argparse` dengan `type=int, default=None` dan help string yang menjelaskan default otomatis
- [x] 1.2 Tambahkan argumen `--model-workers` ke `argparse` dengan `type=int, default=None`
- [x] 1.3 Buat helper function `_default_workers() -> int` yang mengembalikan `min(4, max(1, os.cpu_count() // 2))`
- [x] 1.4 Teruskan nilai `dataset_workers` dan `model_workers` dari CLI ke fungsi `train()` dan `run_batch()`

## 2. Parallel Model Evaluation (`evaluate_models`)

- [x] 2.1 Buat fungsi `_eval_svr(X, y, kfold, log_transform)` yang mengekstrak logika SVR dari `evaluate_models` dan mengembalikan dict partial metrics
- [x] 2.2 Buat fungsi `_eval_lr(X, y, kfold, log_transform)` yang mengekstrak logika LR
- [x] 2.3 Buat fungsi `_eval_rf(X, y, kfold, log_transform)` yang mengekstrak logika RF
- [x] 2.4 Buat fungsi `_eval_xgb(X, y, kfold, log_transform)` yang mengekstrak logika XGB
- [x] 2.5 Refaktor `evaluate_models()` untuk menerima parameter `model_workers: int = 1`
- [x] 2.6 Jika `model_workers > 1`, jalankan `_eval_svr`, `_eval_lr`, `_eval_rf`, `_eval_xgb` secara bersamaan via `ThreadPoolExecutor(max_workers=model_workers)`
- [x] 2.7 Gabungkan hasil dari keempat future ke dalam dict metrics setelah semua selesai
- [x] 2.8 Tetap jalankan LSTM (`train_lstm_with_tuning`) secara sekuensial setelah thread pool selesai
- [x] 2.9 Pastikan `hybrid_weights` dihitung menggunakan MAE dari semua 5 model setelah semua hasil tersedia

## 3. Parallel Dataset Processing (`run_batch`)

- [x] 3.1 Ekstrak body `train_and_export` menjadi fungsi top-level (jika belum) yang bisa di-pickle oleh `ProcessPoolExecutor`
- [x] 3.2 Refaktor `run_batch()` untuk menerima `dataset_workers: int = 1` dan `model_workers: int = 1`
- [x] 3.3 Teruskan `model_workers` ke `train_and_export` → `evaluate_models`
- [x] 3.4 Jika `dataset_workers == 1`, gunakan loop biasa (tanpa executor overhead)
- [x] 3.5 Jika `dataset_workers > 1`, gunakan `ProcessPoolExecutor(max_workers=dataset_workers)` dengan `executor.submit` per CSV path
- [x] 3.6 Tangkap exception per future dengan `try/except` dan log warning tanpa menghentikan future lain
- [x] 3.7 Kumpulkan `trained_stems` dari semua future yang berhasil

## 4. Training Time Logging

- [x] 4.1 Catat `start_time = time.perf_counter()` di awal `train_and_export`
- [x] 4.2 Cetak `[stem] Selesai dalam X.XX detik` di akhir `train_and_export` yang berhasil
- [x] 4.3 Catat `batch_start = time.perf_counter()` di awal `run_batch`
- [x] 4.4 Cetak total wall-clock time, jumlah berhasil, dan jumlah gagal di akhir `run_batch`

## 5. Reproducibility & Seed Isolation

- [x] 5.1 Tambahkan `np.random.seed(42)` di awal `train_and_export` (dipanggil di dalam worker process)
- [x] 5.2 Pastikan `torch.manual_seed(42)` sudah dipanggil di dalam `train_and_export` sebelum semua LSTM training (verifikasi posisi yang ada sudah benar)
- [ ] 5.3 Verifikasi bahwa menjalankan `--dataset-workers 1 --model-workers 1` menghasilkan file `.pkl` dan `_metrics.json` yang identik dengan baseline (bisa via checksum atau perbandingan nilai metrik)

## 6. Verifikasi & Cleanup

- [ ] 6.1 Jalankan `python pipeline/build.py --train --dataset-workers 1 --model-workers 1` dan pastikan tidak ada error
- [ ] 6.2 Jalankan `python pipeline/build.py --train` (default parallel) dan bandingkan output metrik dengan sequential
- [ ] 6.3 Pastikan `python pipeline/build.py --configure` masih berjalan normal (tidak terpengaruh perubahan)
- [x] 6.4 Periksa bahwa tidak ada import baru di luar stdlib yang ditambahkan (hanya `concurrent.futures` dan `time`)

## 7. Upgrade berdasarkan training.txt (746.8s baseline)

- [x] 7.1 Tambahkan `--lstm-workers N` ke argparse; default=None (auto) — untuk ProcessPoolExecutor pada LSTM grid search
- [x] 7.2 Buat fungsi top-level `_lstm_cv_worker(X, y, kfold_n, params, log_transform)` yang bisa di-pickle, wraps `_lstm_cv_scores_with_params`
- [x] 7.3 Refaktor `train_lstm_with_tuning` — tambah `lstm_workers: int = 1`; jika > 1 jalankan 12 combo grid dengan `ProcessPoolExecutor`
- [x] 7.4 Perbaiki `evaluate_models` — SVR+LR saja yang masuk `ThreadPoolExecutor(2)` (mereka tidak saturate cores secara internal); RF+XGB dijalankan sekuensial dengan n_jobs=-1 (manfaatkan internal threading mereka)
- [x] 7.5 Teruskan `lstm_workers` dari `train_and_export` → `evaluate_models` → `train_lstm_with_tuning`
- [x] 7.6 Di `run_batch`: ketika `dataset_workers > 1`, paksa `lstm_workers=1` sebelum masuk worker (hindari nested ProcessPoolExecutor di Windows/spawn)
- [x] 7.7 Teruskan `lstm_workers` dari `train`, `build_all`, dan CLI ke `run_batch`
- [x] 7.8 Default `lstm_workers` di `train()`: jika `dataset_workers == 1`, auto = `_default_workers()`; jika `dataset_workers > 1`, forced = 1
