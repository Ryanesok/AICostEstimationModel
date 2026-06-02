## 1. Setup & Dependencies

- [x] 1.1 Tambahkan `requests` ke `requirements.txt`
- [x] 1.2 Jalankan `pip install -r requirements.txt` untuk memastikan `requests` terinstall
- [x] 1.3 Hapus file `.pkl` lama (`svr_model.pkl`, `lr_model.pkl`, `scaler.pkl`) dari `models/` yang menggunakan skema penamaan lama

## 2. Dataset Downloader

- [x] 2.1 Buat `datasets.txt` berisi minimal dua URL raw CSV dari PROMISE repository (COCOMO81 & Desharnais); tambahkan komentar penjelasan di baris pertama
- [x] 2.2 Buat `downloader.py` — baca `datasets.txt`, skip baris kosong dan komentar (`#`)
- [x] 2.3 Implementasikan logika download per URL: ekstrak nama file dari URL, simpan ke `data/<filename>`
- [x] 2.4 Tambahkan cek file-exists (skip jika sudah ada) dan try/except per URL (log WARNING, lanjut ke URL berikutnya)
- [x] 2.5 Tambahkan blok `if __name__ == "__main__"` dengan ringkasan hasil (berapa file diunduh, berapa di-skip, berapa gagal)

## 3. Multi-Dataset Training Pipeline (`model_pipeline.py`)

- [x] 3.1 Definisikan `DATASET_CONFIG` dict — tambahkan entry untuk `cocomo81` (features & target) dan `desharnais` (features & target)
- [x] 3.2 Ubah fungsi `load_or_generate_data()` menjadi `load_dataset(csv_path)` yang hanya memuat CSV (tanpa dummy-data fallback)
- [x] 3.3 Implementasikan `get_config(stem)` — lookup `DATASET_CONFIG` by stem, return config atau `None` jika tidak dikenal
- [x] 3.4 Implementasikan `train_and_export(csv_path)` — load CSV, get config, preprocess (drop NaN + StandardScaler + train/test split), train SVR & LR, evaluate & print MAE+R2, export `svr_<stem>.pkl` + `linreg_<stem>.pkl` + `scaler_<stem>.pkl`
- [x] 3.5 Implementasikan `run_batch()` — glob semua `data/*.csv`, iterasi, panggil `train_and_export()` per file; log warning jika `data/` kosong
- [x] 3.6 Wire up `if __name__ == "__main__"` untuk memanggil `run_batch()` dan cetak ringkasan total model yang berhasil dibuat

## 4. Dynamic GUI (`app.py`)

- [x] 4.1 Implementasikan `discover_models()` — glob `models/svr_*.pkl`, ekstrak stem list, return sorted list atau empty list
- [x] 4.2 Implementasikan `load_model_trio(stem)` — load `svr_<stem>.pkl`, `linreg_<stem>.pkl`, `scaler_<stem>.pkl`, return tuple atau error string
- [x] 4.3 Update `_build_ui()` — tambahkan `CTkOptionMenu` (dataset selector) di atas form; jika `discover_models()` kosong, tampilkan error banner
- [x] 4.4 Implementasikan callback `_on_dataset_change(choice)` — panggil `load_model_trio(choice)`, reset form dan output, update state
- [x] 4.5 Pertahankan `CTkSegmentedButton` SVR/LR toggle — sekarang toggle di dalam dataset yang dipilih
- [x] 4.6 Tambahkan section kalkulator di bawah hasil estimasi: dua CTkEntry (gaji bulanan Rupiah, target durasi bulan) + dua CTkLabel output (Total Biaya, Kebutuhan Tim)
- [x] 4.7 Implementasikan `_update_calculator()` — Total Biaya = prediksi × gaji; Kebutuhan Tim = ceil(prediksi / durasi); panggil otomatis setelah estimasi berhasil dan saat input kalkulator berubah
- [x] 4.8 Update `_on_reset()` — tambahkan clearing input kalkulator (gaji, durasi) dan output kalkulator (biaya, tim)
- [x] 4.9 Wire up `if __name__ == "__main__"` — jalankan `discover_models()` dan launch app

## 6. Konfigurasi Otomatis & Pemisahan Config

- [x] 6.1 Tambahkan `pyyaml` ke `requirements.txt` dan install
- [x] 6.2 Buat `dataset_config.yaml` — pindahkan semua entry dari `DATASET_CONFIG` di `model_pipeline.py` ke format YAML; tambahkan entry `02.desharnais` berdasarkan kolom CSV yang ada
- [x] 6.3 Update `model_pipeline.py` — hapus `DATASET_CONFIG` hardcoded, tambahkan `load_config()` yang membaca `dataset_config.yaml`; perbarui `get_config()` agar pakai config yang dimuat
- [x] 6.4 Buat `auto_configure.py` — scan `data/*.csv`, deteksi kolom numerik, skip dataset yang sudah dikonfigurasi, prompt user untuk memilih kolom target, simpan entry baru ke `dataset_config.yaml`
- [x] 6.5 Update `app.py` — tambahkan `load_field_config_from_yaml(stem)` sebagai fallback saat stem tidak ada di `FIELD_CONFIGS`; gunakan nama kolom asli sebagai label

- [x] 6.6 `model_pipeline.py` — tambahkan `cleanup_stale_models(trained_stems)`: setelah `run_batch()`, glob semua `svr_*.pkl` di `models/`, hapus trio (svr, linreg, scaler) yang stemnya tidak ada dalam set dataset yang berhasil ditraining; cetak ringkasan file yang dihapus
- [x] 6.7 `auto_configure.py` — tambahkan cleanup konfigurasi usang: setelah scan selesai, deteksi entry di YAML yang tidak memiliki CSV di `data/`, tampilkan daftar dan minta konfirmasi user untuk menghapusnya sebelum menyimpan

- [x] 6.8 `app.py` — tambahkan entry `cocomo-81` ke `FIELD_CONFIGS` dengan label dan hint Bahasa Indonesia lengkap untuk semua 16 kolom COCOMO cost driver (rely, data, cplx, time, stor, virt, turn, acap, aexp, pcap, vexp, lexp, modp, tool, sced, loc)
- [x] 6.9 `app.py` — perbaiki error "invalid command name" saat window ditutup: override `WM_DELETE_WINDOW` dengan `_on_close()` yang menutup semua matplotlib figure lalu memanggil `quit()` + `destroy()`

- [x] 6.10 `downloader.py` — `arff_to_csv()` + deteksi `.arff` di `download_file()` terverifikasi — tambahkan `arff_to_csv(content_bytes, dest_path)`: parse header `@attribute` sebagai nama kolom CSV, ambil baris setelah `@data`, ganti `?` (missing value ARFF) dengan string kosong; update `download_file()` untuk mendeteksi URL berekstensi `.arff`, konversi ke CSV, simpan dengan nama file `.csv`

- [x] 6.11 Extend `dataset_config.yaml` — tambahkan section `labels` dan `hints` per dataset; migrasikan semua data dari `FIELD_CONFIGS` di `app.py` untuk `cocomo-81` dan `desharnais`
- [x] 6.12 Update `app.py` — hapus `FIELD_CONFIGS` dict; update `load_field_config_from_yaml(stem)` agar membaca `labels` dan `hints` dari YAML; gunakan `labels.get(col, col)` sebagai label dan `hints.get(col, "")` sebagai hint
- [x] 6.13 Update `auto_configure.py` — setelah user konfirmasi features dan target, tawarkan pengisian `labels` dan `hints` per kolom secara interaktif (tekan Enter untuk melewati); simpan ke entry YAML

## 7. Konversi Satuan Effort (Person-Hours → Person-Months)

- [x] 7.1 `dataset_config.yaml` — tambahkan field `effort_unit` per dataset: `"person-months"` untuk `cocomo-81`, `"person-hours"` untuk `desharnais`, `china`, dan `maxwell`; tambahkan komentar penjelasan di header file
- [x] 7.2 `app.py` — tambahkan fungsi `get_effort_unit(stem)`: baca `effort_unit` dari YAML, return `"person-hours"` atau `"person-months"` (default `"person-months"` jika tidak ada)
- [x] 7.3 `app.py` — tambahkan konstanta `HOURS_PER_MONTH = 160` dan fungsi `to_person_months(value, unit)`: jika `unit == "person-hours"` bagi dengan `HOURS_PER_MONTH`, else return value as-is
- [x] 7.4 `app.py` — update `_on_estimate()`: setelah prediksi, ambil unit dataset, konversi ke person-months, simpan nilai terkonversi ke `_pred_svr`/`_pred_lr`; update label unit agar menampilkan unit asli dan hasil konversi (contoh: `"Person-Months (dari person-hours ÷ 160) [SVR]"`)
- [x] 7.5 `app.py` — update `_on_model_toggle()`: tampilkan label unit yang sama (dengan info konversi jika relevan) setelah toggle model
- [x] 7.6 `app.py` — update `_update_chart()`: ganti label sumbu X menjadi `"Person-Months"` dan tambahkan subtitle `"(dikonversi dari person-hours)"` jika dataset berbasis person-hours

## 8. Slider Inputs & Labels untuk China/Maxwell

- [x] 8.1 `dataset_config.yaml` — tambahkan `sliders` dict untuk `cocomo-81`: 15 cost driver dengan `values` list discrete (snap); `loc` dengan `min/max/step/default` continuous
- [x] 8.2 `dataset_config.yaml` — tambahkan `sliders` dict untuk `desharnais` (TeamExp, ManagerExp, Length) dan `maxwell` (Syear, App, Har, Dba, Ifc, Source, Telonuse, Nlan, T01-T15, Time — semua integer bounded)
- [x] 8.3 `dataset_config.yaml` — tambahkan `labels` dan `hints` untuk `china` (17 fitur: AFP, Input/Output/Enquiry/File/Interface, Added/Changed/Deleted, PDR/NPDR FP, Resource, Dev.Type, N_effort, Effort)
- [x] 8.4 `dataset_config.yaml` — tambahkan `labels` dan `hints` untuk `maxwell` (26 fitur: Syear, App, Har, Dba, Ifc, Source, Telonuse, Nlan, T01-T15, Size, Time, Effort)
- [x] 8.5 `app.py` — tambahkan `load_slider_config_from_yaml(stem)`: baca `sliders` dict dari YAML, return `{col: config_dict}`
- [x] 8.6 `app.py` — tambahkan `_field_readers: dict` dan `_field_resetters: dict` di `__init__`; update `_rebuild_form()` agar field dengan slider config menggunakan `CTkSlider` di subframe + `CTkLabel` nilai; untuk `values` list: snap ke item terdekat; untuk `min/max/step`: slider continuous; keduanya update `_field_readers[key]` dan `_field_resetters[key]`
- [x] 8.7 `app.py` — update `_on_estimate()` untuk membaca nilai via `float(self._field_readers[key]())`; update `_on_reset()` untuk memanggil semua `_field_resetters[key]()`

## 5. Verification

- [x] 5.1 Jalankan `python downloader.py` — konfirmasi minimal satu CSV terunduh ke `data/`, jalankan kedua kali dan konfirmasi file di-skip
- [x] 5.2 Jalankan `python model_pipeline.py` — konfirmasi model `.pkl` dibuat dengan format nama `svr_<stem>.pkl` per dataset
- [x] 5.3 Jalankan `python model_pipeline.py` di folder `data/` kosong — konfirmasi warning muncul dan tidak crash
- [x] 5.4 Jalankan `python app.py` — konfirmasi CTkOptionMenu terisi dataset yang tersedia
- [x] 5.5 Pilih dataset berbeda dari dropdown — konfirmasi form direset dan model yang dimuat berubah
- [x] 5.6 Isi form dan klik "Estimasi" — konfirmasi prediksi tampil dan kalkulator otomatis menghitung biaya & tim
- [x] 5.7 Klik "Reset Form" — konfirmasi semua field, output estimasi, dan output kalkulator bersih
