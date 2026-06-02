## 1. RF Hyperparameter Tuning

- [x] 1.1 Ganti blok training RF di `model_pipeline.py` dengan `RandomizedSearchCV` menggunakan param grid `n_estimators`, `max_depth`, `min_samples_split` dan `n_iter=30`, scorer `neg_mean_absolute_error`, `cv=5`
- [x] 1.2 Simpan `best_params_rf` ke dict metrics sebelum ditulis ke `_metrics.json`
- [x] 1.3 Verifikasi `_metrics.json` salah satu dataset memiliki key `best_params_rf` setelah training ulang

## 2. LSTM Hyperparameter Grid Search

- [x] 2.1 Definisikan grid `hidden_size` ∈ [32, 64, 128], `num_layers` ∈ [1, 2], `learning_rate` ∈ [0.001, 0.0005] di `model_pipeline.py`
- [x] 2.2 Implementasi loop grid search: untuk setiap kombinasi, jalankan 5-fold CV dan hitung mean MAE
- [x] 2.3 Pilih kombinasi dengan MAE terkecil, retrain LSTM final pada full training set dengan hyperparameter tersebut
- [x] 2.4 Simpan `best_params_lstm` ke dict metrics sebelum ditulis ke `_metrics.json`
- [x] 2.5 Verifikasi `_metrics.json` memiliki key `best_params_lstm` setelah training ulang

## 3. Backward Compatibility Metrics

- [x] 3.1 Pastikan semua pembacaan `best_params_rf` dan `best_params_lstm` di kode menggunakan `.get("best_params_rf", {})` sehingga file lama tidak `KeyError`

## 4. Export Hasil Estimasi

- [x] 4.1 Tambah tombol "Export" di panel hasil `app.py`, disabled secara default
- [x] 4.2 Setelah estimasi berhasil, enable tombol Export dan simpan state `last_result` (inputs + predictions)
- [x] 4.3 Implementasi handler tombol: buka `filedialog.asksaveasfilename` dengan filter `.csv` dan `.json`
- [x] 4.4 Implementasi fungsi `export_csv(path, inputs, predictions)` — tulis header + satu data row + kolom `timestamp`
- [x] 4.5 Implementasi fungsi `export_json(path, inputs, predictions)` — tulis objek `{"inputs": ..., "predictions": ..., "timestamp": ...}`
- [x] 4.6 Tangkap exception `OSError`/`IOError` dari operasi tulis dan tampilkan pesan error di GUI
- [x] 4.7 Uji export CSV dan JSON: buka file hasil, verifikasi semua kolom/key hadir dan nilainya benar

## 5. Perbandingan Model Side-by-Side

- [x] 5.1 Tambah `CTkScrollableFrame` (atau frame biasa) di panel hasil berisi satu `CTkCheckBox` per model
- [x] 5.2 Set semua checkbox ke checked setelah estimasi selesai; cegah uncheck jika hanya tersisa satu yang aktif
- [x] 5.3 Buat fungsi `render_comparison_table(selected_models)` yang merender grid `CTkLabel` dengan baris: Prediction, MAE, RMSE, R² dan kolom: nama model
- [x] 5.4 Bind setiap checkbox ke `render_comparison_table` + `render_comparison_chart` agar tabel dan chart terupdate saat pilihan berubah
- [x] 5.5 Buat fungsi `render_comparison_chart(selected_models)` — bar chart prediksi, label model di-truncate ke 10 karakter
- [ ] 5.6 Uji: tambah/hapus checkbox dan verifikasi tabel + chart terupdate tanpa error

## 6. Persistensi Input Form

- [x] 6.1 Tambah fungsi `save_last_inputs(dataset_name, field_values)` yang menulis `last_inputs/<dataset_name>.json`; buat direktori `last_inputs/` bila belum ada
- [x] 6.2 Panggil `save_last_inputs` setelah estimasi berhasil dijalankan
- [x] 6.3 Tambah fungsi `load_last_inputs(dataset_name)` yang membaca file JSON; kembalikan `{}` bila tidak ada atau parse gagal
- [x] 6.4 Panggil `load_last_inputs` saat dataset dipilih dan form diinisialisasi; isi field dengan nilai yang dikembalikan
- [x] 6.5 Tambah `last_inputs/` ke `.gitignore`
- [ ] 6.6 Uji: isi form, run estimasi, restart app, pilih dataset yang sama — verifikasi field terisi nilai sebelumnya
