## Why

`pipeline/build.py` processes setiap dataset secara **sekuensial** dan mengevaluasi setiap model secara **sekuensial** di dalam setiap dataset — kombinasi ini menyebabkan waktu training skala linier terhadap jumlah dataset dan jumlah model. Dengan dataset kecil-menengah (puluhan hingga ratusan baris) dan 5 model (SVR, LR, RF, XGB, LSTM), sebagian besar CPU cores dibiarkan idle selama proses berlangsung.

## What Changes

- **Parallel dataset processing**: `run_batch()` akan memproses beberapa dataset secara paralel menggunakan `ProcessPoolExecutor`, bukan loop sekuensial.
- **Parallel model evaluation**: di dalam `evaluate_models()`, GridSearch/CV untuk SVR, LR, RF, XGB, dan LSTM akan dijalankan secara konkuren menggunakan `ThreadPoolExecutor` (aman karena tiap estimator sudah melepas GIL via numpy/C extensions).
- **Configurable worker count**: jumlah worker dataset-level dan model-level dapat dikonfigurasi via argumen CLI (`--dataset-workers`, `--model-workers`) dengan default otomatis berdasarkan `os.cpu_count()`.
- **Progress reporting**: menambahkan output ringkasan waktu per dataset dan total waktu training agar pengguna dapat mengukur speedup.
- Tidak ada perubahan pada logika model, hyperparameter grid, metrik evaluasi, atau format output artifact (`.pkl`, `.pt`, `.json`).

## Capabilities

### New Capabilities

- `parallel-training`: Kemampuan menjalankan training beberapa dataset secara bersamaan dan mengevaluasi beberapa model secara bersamaan di dalam satu dataset, dengan fallback sequential jika workers=1.

### Modified Capabilities

- `pipeline-module`: Requirement `run_batch()` berubah — eksekusi tidak lagi harus sekuensial; output dan artifact yang dihasilkan tetap sama.

## Impact

- **`pipeline/build.py`**: Refaktor `run_batch()` dan `evaluate_models()` — satu-satunya file yang berubah.
- **Dependensi**: Tidak ada dependensi baru; `concurrent.futures` tersedia di stdlib Python 3.9+.
- **Reproducibility**: `random_state=42` dan `torch.manual_seed(42)` per worker menjaga hasil deterministik (seed di-set di dalam setiap worker process).
- **Memory**: Peak memory naik proporsional dengan jumlah dataset workers aktif secara bersamaan — default dibatasi `min(4, cpu_count // 2)` untuk mencegah OOM pada mesin dengan RAM terbatas.
- **LSTM thread-safety**: LSTM training menggunakan `ThreadPoolExecutor` (bukan process), karena PyTorch + model state tidak aman di-fork lintas proses tanpa serialisasi eksplisit.
