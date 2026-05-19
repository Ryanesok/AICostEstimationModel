## 1. Dependencies

- [x] 1.1 `requirements.txt` — tambahkan `xgboost` dan `torch` (CPU-only install note di komentar)

## 2. `model_pipeline.py` — Fungsi Training Baru

- [x] 2.1 `model_pipeline.py` — tambahkan import: `from sklearn.ensemble import RandomForestRegressor` dan `from xgboost import XGBRegressor`
- [x] 2.2 `model_pipeline.py` — tambahkan import torch: `import torch`, `import torch.nn as nn`, `from torch.utils.data import DataLoader, TensorDataset`
- [x] 2.3 `model_pipeline.py` — tambahkan konstanta `XGB_GRID_PARAMS` di bagian atas: `{"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}`
- [x] 2.4 `model_pipeline.py` — buat fungsi `train_rf(X, y)` yang melatih `RandomForestRegressor(n_estimators=200, random_state=42)` pada seluruh X, mengembalikan model
- [x] 2.5 `model_pipeline.py` — buat fungsi `train_xgb(X, y, kfold)` yang menjalankan `GridSearchCV(XGBRegressor(...), XGB_GRID_PARAMS, cv=kfold)`, fit pada seluruh X, mengembalikan `(best_estimator, best_params)`
- [x] 2.6 `model_pipeline.py` — buat kelas `LSTMModel(nn.Module)` dengan satu LSTM layer (hidden=32) dan satu linear output layer
- [x] 2.7 `model_pipeline.py` — buat fungsi `train_lstm(X, y, input_size, epochs=200, patience=20)` yang melatih `LSTMModel` dengan Adam + MSE, early stopping pada 20% validation split, mengembalikan trained model

## 3. `model_pipeline.py` — Integrasi ke `evaluate_models()` dan `train_and_export()`

- [x] 3.1 `model_pipeline.py` — update `evaluate_models(X, y)` agar menghitung CV metrics (mae, r2) untuk RF, XGBoost, dan LSTM menggunakan KFold yang sama; tambahkan `rf_cv_mae_mean`, `rf_cv_r2_mean`, `xgb_cv_mae_mean`, `xgb_cv_r2_mean`, `lstm_cv_mae_mean`, `lstm_cv_r2_mean` ke dict hasil
- [x] 3.2 `model_pipeline.py` — update `evaluate_models()` agar menghitung `hybrid_weights: {"lstm": w1, "xgb": w2, "lr": w3}` berdasarkan inverse CV MAE (dengan floor `1e-6`)
- [x] 3.3 `model_pipeline.py` — update `train_and_export()` agar memanggil `train_rf()`, `train_xgb()`, `train_lstm()` setelah SVR/LR selesai; simpan hasil ke pkl/pt
- [x] 3.4 `model_pipeline.py` — simpan `rf_<stem>.pkl` dan `xgb_<stem>.pkl` via joblib; simpan `lstm_<stem>.pt` via `torch.save(model.state_dict(), ...)` dan `lstm_arch_<stem>.json` via json
- [x] 3.5 `model_pipeline.py` — update print summary di `train_and_export()` agar mencetak metrics RF, XGBoost, LSTM, dan hybrid weights

## 4. `model_pipeline.py` — Cleanup

- [x] 4.1 `model_pipeline.py` — update `cleanup_stale_models()` agar juga menghapus `rf_*.pkl`, `xgb_*.pkl`, `lstm_*.pt`, `lstm_arch_*.json` untuk stem yang usang

## 5. `app.py` — Load Model Baru

- [x] 5.1 `app.py` — tambahkan import `torch` dan definisikan kelas `LSTMModel` yang identik dengan yang di `model_pipeline.py` (atau import dari modul bersama)
- [x] 5.2 `app.py` — tambahkan `import json` jika belum ada; tambahkan field `self._rf`, `self._xgb`, `self._lstm`, `self._hybrid_weights` di `__init__`
- [x] 5.3 `app.py` — buat fungsi `_load_lstm(stem)` yang membaca `lstm_arch_<stem>.json`, membuat `LSTMModel(input_size)`, memuat `state_dict` dari `lstm_<stem>.pt`, set `eval()`, mengembalikan model atau `None` jika file tidak ada
- [x] 5.4 `app.py` — update `_load_dataset(stem)` agar juga memuat `rf_<stem>.pkl`, `xgb_<stem>.pkl`, LSTM via `_load_lstm()`, dan `hybrid_weights` dari `<stem>_metrics.json`

## 6. `app.py` — Model Selector

- [x] 6.1 `app.py` — di `_build_middle_panel()`, ganti `CTkSegmentedButton` dengan `CTkOptionMenu` menggunakan nilai `["SVR", "Linear Regression", "Random Forest", "XGBoost", "LSTM", "Hybrid"]` dan `self._model_var`
- [x] 6.2 `app.py` — tambahkan `command` pada `CTkOptionMenu` yang memanggil `self._on_model_change(choice)` untuk memperbarui highlight chart saat model berganti tanpa klik Estimasi ulang

## 7. `app.py` — Estimasi dan Reset

- [x] 7.1 `app.py` — update `_on_estimate()`: tambahkan dispatch ke RF (`self._rf.predict`), XGBoost (`self._xgb.predict`), LSTM (forward pass), dan Hybrid (weighted sum); simpan semua 6 prediksi ke `self._all_preds: dict[str, float]`
- [x] 7.2 `app.py` — update `_on_estimate()` agar menampilkan nilai prediksi dari model yang dipilih (`self._model_var.get()`) di result label
- [x] 7.3 `app.py` — update `_on_reset()` agar membersihkan `self._all_preds` dan mereset chart

## 8. `app.py` — Chart dengan Highlight

- [x] 8.1 `app.py` — update `_draw_chart()` agar menerima `all_preds: dict[str, float]` (semua 6 prediksi) dan `selected: str` (nama model terpilih)
- [x] 8.2 `app.py` — di `_draw_chart()`, render 6 bar horizontal; bar untuk `selected` menggunakan warna `#4a9eff`; bar lainnya `#555555`
- [x] 8.3 `app.py` — update pemanggilan `_draw_chart()` di `_on_estimate()` dan `_on_model_change()` agar meneruskan `all_preds` dan model terpilih

## 9. Verifikasi

- [x] 9.1 Jalankan `pip install xgboost torch` dan konfirmasi instalasi sukses
- [x] 9.2 Jalankan `python model_pipeline.py` — konfirmasi semua 4 dataset menghasilkan `rf_*.pkl`, `xgb_*.pkl`, `lstm_*.pt`, `lstm_arch_*.json`, dan metrics JSON diperbarui
- [ ] 9.3 Jalankan `python app.py` — konfirmasi dropdown model menampilkan 6 opsi
- [ ] 9.4 Pilih dataset, isi form, klik Estimasi — konfirmasi hasil tampil; ganti model di dropdown — konfirmasi nilai dan chart highlight berubah
- [ ] 9.5 Klik Reset Form — konfirmasi chart kembali kosong dan dropdown kembali ke SVR
