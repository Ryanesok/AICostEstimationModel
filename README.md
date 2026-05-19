# AI Cost Estimation Model

Aplikasi desktop untuk estimasi effort pengembangan perangkat lunak menggunakan machine learning. Mendukung enam model prediksi (SVR, Linear Regression, Random Forest, XGBoost, LSTM, Hybrid) pada empat dataset historis yang telah terbukti di riset rekayasa perangkat lunak.

---

## Prerequisites

- Python 3.11 atau lebih baru
- pip (biasanya sudah tersedia bersama Python)
- Dependensi utama (diinstall otomatis via `requirements.txt`):

| Paket | Kegunaan |
|---|---|
| `customtkinter` | GUI framework |
| `scikit-learn` | SVR, Linear Regression, Random Forest |
| `xgboost` | XGBoost model |
| `torch` (CPU) | LSTM model |
| `pandas` / `numpy` | Data processing |
| `matplotlib` | Chart visualisasi |
| `pyyaml` | Konfigurasi dataset |
| `joblib` | Serialisasi model |

---

## Setup & Cara Menjalankan

### Langkah 1 — Install dependensi

```bash
pip install -r requirements.txt
# Untuk torch CPU-only (lebih ringan):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Langkah 2 — Download dataset

```bash
python downloader.py
```

Mengunduh empat file CSV ke folder `data/`: COCOMO-81, Desharnais, China, Maxwell.

### Langkah 3 — Konfigurasi dataset (opsional)

```bash
python auto_configure.py
```

Mendeteksi kolom secara otomatis, memilih target, dan memeriksa kebocoran fitur. Hasilnya disimpan di `dataset_config.yaml`. Langkah ini sudah memiliki default yang baik — lewati jika tidak ada kustomisasi.

### Langkah 4 — Latih model

```bash
python model_pipeline.py
```

Melatih semua enam model untuk setiap dataset menggunakan 5-fold cross-validation. Model disimpan ke folder `models/`.

### Langkah 5 — Jalankan aplikasi

```bash
python app.py
```

Membuka antarmuka grafis. Pilih dataset, isi form fitur, klik **Estimasi**.

---

## Model yang Tersedia

| Model | Keterangan | Tuning |
|---|---|---|
| **SVR** | Support Vector Regression | GridSearchCV (C, epsilon, kernel) |
| **Linear Regression** | Regresi linier OLS | — |
| **Random Forest** | Ensemble 200 pohon | `n_estimators=200, random_state=42` |
| **XGBoost** | Gradient boosting | GridSearchCV (n_estimators, max_depth, learning_rate) |
| **LSTM** | Recurrent neural network 1 layer | Adam + MSE, early stopping (patience=20) |
| **Hybrid** | Rata-rata berbobot LSTM + XGBoost + LR | Bobot = inverse CV MAE, dinormalisasi |

Semua model dievaluasi dengan 5-fold CV. Metrik tersimpan di `models/<dataset>_metrics.json`.

---

## File Layout

```
AICostEstimationModel/
│
├── app.py                   # Antarmuka GUI (CustomTkinter)
├── model_pipeline.py        # Training + export semua model
├── auto_configure.py        # Deteksi otomatis kolom & konfigurasi
├── downloader.py            # Mengunduh dataset CSV
│
├── dataset_config.yaml      # Konfigurasi fitur, target, label, hint per dataset
├── field_labels.yaml        # Default label & hint (fallback auto_configure)
├── datasets.txt             # Daftar URL dataset untuk downloader
├── requirements.txt         # Dependensi Python
│
├── data/                    # Dataset CSV (dihasilkan oleh downloader.py)
│   ├── COCOMO-81.csv
│   ├── Desharnais.csv
│   ├── china.csv
│   └── maxwell.csv
│
├── models/                  # Model terlatih (dihasilkan oleh model_pipeline.py)
│   ├── svr_<dataset>.pkl
│   ├── linreg_<dataset>.pkl
│   ├── rf_<dataset>.pkl
│   ├── xgb_<dataset>.pkl
│   ├── scaler_<dataset>.pkl
│   ├── lstm_<dataset>.pt
│   ├── lstm_arch_<dataset>.json
│   └── <dataset>_metrics.json
│
├── docs/                    # Panduan pengisian field per dataset
│   ├── cocomo-81.md
│   ├── desharnais.md
│   ├── china.md
│   └── maxwell.md
│
└── openspec/                # Dokumentasi perubahan (OpenSpec workflow)
```

---

## Panduan Field Dataset

Lihat folder `docs/` untuk panduan lengkap per dataset:

- [docs/cocomo-81.md](docs/cocomo-81.md) — 16 field, effort dalam person-months
- [docs/desharnais.md](docs/desharnais.md) — 5 field, effort dalam person-hours
- [docs/china.md](docs/china.md) — 14 field, effort dalam person-hours
- [docs/maxwell.md](docs/maxwell.md) — 25 field, effort dalam person-hours
