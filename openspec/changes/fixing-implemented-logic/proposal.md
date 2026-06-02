## Why

The current `model_pipeline.py` and `auto_configure.py` were built for speed-to-result: single train/test split, default hyperparameters, no validation of feature quality, and metrics that are printed but never saved. On small datasets (COCOMO-81: ~63 rows, Desharnais: ~77 rows), this produces unreliable model performance estimates and risks silent data leakage — harming correctness of predictions and blocking confident future development.

## What Changes

- **Cross-validation in training**: Replace single `train_test_split(test_size=0.2)` with k-fold cross-validation (k=5) to produce stable MAE/R² estimates on small datasets; keep a held-out test set for final reporting
- **SVR hyperparameter tuning**: Replace hardcoded `SVR(kernel="rbf")` with a `GridSearchCV` over `C`, `epsilon`, and `kernel` to find parameters that actually fit each dataset
- **Feature leakage guard in `auto_configure.py`**: Warn the user (and require confirmation) when a candidate feature is highly correlated (|r| > 0.95) with the selected target — indicates near-duplicate or derived columns
- **Feature correlation filter**: Optionally drop one of any pair of features with |r| > 0.90 to reduce multicollinearity before training
- **Evaluation export**: Save per-dataset metrics (CV MAE, CV R², test MAE, test R², best SVR params) to `models/<stem>_metrics.json` alongside the `.pkl` files
- **Data quality checks**: Validate minimum row count before splitting (raise clear error if < 20 rows); detect and report constant/near-constant columns in `auto_configure.py`

## Capabilities

### New Capabilities

- `cross-validated-training`: Training pipeline uses k-fold CV + final test evaluation, saves metrics JSON per dataset
- `svr-hyperparameter-tuning`: SVR trained via GridSearchCV; best params serialized into metrics JSON
- `feature-leakage-guard`: `auto_configure.py` computes target correlation for each candidate feature and warns before saving config

### Modified Capabilities

- none

## Impact

- `model_pipeline.py`: `preprocess()`, `train_models()`, `evaluate()`, `train_and_export()` — logic changes, public signatures may change slightly
- `auto_configure.py`: `configure_one()`, `collect_labels_hints()` — adds correlation check step after target selection
- `models/` directory: new `<stem>_metrics.json` files created alongside existing `.pkl` files
- `app.py`: no changes required — model loading and prediction API (`svr.predict`, `lr.predict`) unchanged

---

## Temuan Bug Tambahan (Ditemukan Pasca-Implementasi)

Bug berikut ditemukan saat pengujian runtime aplikasi setelah implementasi tahap pertama selesai. Keduanya berdampak langsung pada hasil estimasi yang ditampilkan ke user.

### Bug #1 — `select_best_estimator` selalu memilih isbsg10/XGBoost

**Root cause:** `select_best_estimator` di `pipeline/estimator.py` memilih model berdasarkan `cv_mae_pm` (MAE dinormalisasi ke person-months). Dataset yang menggunakan satuan `person-hours` mendapat keuntungan artifisial karena MAE-nya dibagi 160 sebelum dibandingkan.

**Bukti dari metrics yang tersimpan:**

| Dataset | Model | cv_mae (native) | Satuan | cv_mae_pm |
|---|---|---|---|---|
| **isbsg10** | **XGBoost** | **882** | person-hours | **5.52 pm ← selalu menang** |
| kitchenham | RF | 1,275 | person-hours | 7.97 pm |
| kitchenham | LSTM | 1,280 | person-hours | 8.01 pm |
| kemerer | LSTM | 82 | person-months | 82 pm |
| desharnais | semua | ~2,300 | person-hours | ~14 pm |

isbsg10 selalu unggul bukan karena kualitasnya lebih baik, melainkan karena nilai target `N_effort`-nya secara absolut kecil dalam person-hours. **isbsg10 hanya memiliki 37 baris data training** — terlalu sedikit untuk XGBoost sehingga tidak bisa mengekstrapolasi di luar rentang training. Untuk input besar (misal 100 fitur, high complexity), model ini tetap memprediksi nilai yang capped di batas atas training data (~4000 person-hours = 25 pm).

**Dampak:** Prediksi effort konsisten terlalu kecil (sekitar 25 pm) untuk semua variasi input besar karena XGBoost tidak bisa ekstrapolasi.

**Solusi:** Tambahkan threshold minimum jumlah baris training di `select_best_estimator` (minimal 50 baris) agar dataset kecil seperti isbsg10 (37 baris) tidak dapat memenangkan seleksi.

---

### Bug #2 — `team_needed` selalu bernilai 1

**Root cause:** Formula di `core/cost_calculator.py`:

```python
team_needed = math.ceil(effort_pm / max(deadline_months, 0.1))
```

Ketika deadline dimaksimalkan (60 bulan dari slider), hampir semua nilai `effort_pm` yang reasonable menghasilkan nilai < 1 sebelum dibulatkan ke atas:

```
ceil(25 pm / 60 bulan) = ceil(0.42) = 1
ceil(50 pm / 60 bulan) = ceil(0.83) = 1
ceil(60 pm / 60 bulan) = ceil(1.00) = 1
```

Selain itu, field `num_developers` yang diinput user di wizard **sama sekali tidak dipakai** dalam perhitungan `team_needed` — hanya digunakan untuk `cost_per_developer`.

**Dampak:** Berapapun jumlah fitur, kompleksitas, atau tim yang diinput, `team_needed` selalu = 1 selama deadline cukup panjang. Surplus anggaran tampak sangat besar karena effort rendah × salary menghasilkan biaya kecil vs budget default besar.

**Bukti trace dengan input maksimum:**
```
effort_pm ≈ 25 pm  (isbsg10/XGB terkunci di batas training data)
team_needed = ceil(25 / 60) = 1
total_cost = 25 × Rp 20.000.000 = Rp 500.000.000
surplus = Rp 2.000.000.000 − Rp 500.000.000 = Rp 1.500.000.000 (75%)
```

**Solusi:** Pisahkan kalkulasi menjadi dua nilai:
- `min_team_for_deadline`: minimum developer agar selesai tepat waktu (formula lama)
- `actual_duration_months`: durasi aktual jika menggunakan `num_developers` yang diinput user (`effort_pm / num_developers`)

Tampilkan keduanya di UI sehingga user mendapat konteks yang lebih berguna.
