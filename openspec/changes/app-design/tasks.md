## 1. Window Size

- [x] 1.1 `app.py` — ubah `self.geometry("1000x780")` menjadi `"1400x780"`

## 2. Rewrite `_build_ui()` — Grid 3 Kolom

- [x] 2.1 `app.py` — ganti konfigurasi grid di `_build_ui()`: `grid_columnconfigure(0, weight=0, minsize=420)`, `grid_columnconfigure(1, weight=0, minsize=380)`, `grid_columnconfigure(2, weight=1)`; hapus konfigurasi kolom lama
- [x] 2.2 `app.py` — buat `middle = CTkFrame(self, corner_radius=12)` dan tempatkan di `row=0, column=1, padx=(8,8), pady=16, sticky="nsew"`; simpan sebagai `self._middle`; panggil `self._build_middle_panel(middle)` setelah form/error banner
- [x] 2.3 `app.py` — update chart frame dari `column=1` ke `column=2`; update `padx` dari `(8, 16)` tetap ke `(8, 16)` (kiri/kanan panel kanan)

## 3. Buat `_build_middle_panel(parent)`

- [x] 3.1 `app.py` — buat method `_build_middle_panel(parent)`: tambahkan heading `"Model & Hasil Estimasi"` di `row=0`; konfigurasi `parent.grid_columnconfigure(1, weight=1)`
- [x] 3.2 `app.py` — pindahkan model toggle (`CTkLabel "Model Prediksi"` + `CTkSegmentedButton`) ke `_build_middle_panel()` di row=1; buat `self._model_var` di sini
- [x] 3.3 `app.py` — pindahkan `self._error_label` ke `_build_middle_panel()` di row=2
- [x] 3.4 `app.py` — pindahkan `btn_frame` (tombol Estimasi + Reset Form) ke `_build_middle_panel()` di row=3; jika `self._stems` kosong, set `state="disabled"` pada tombol Estimasi
- [x] 3.5 `app.py` — pindahkan result area (`CTkFrame` + `self._result_label` + `self._unit_label`) ke `_build_middle_panel()` di row=4; `padx=16`
- [x] 3.6 `app.py` — pindahkan calculator section (`self._gaji_entry`, `self._durasi_entry`, `self._biaya_label`, `self._tim_label`) ke `_build_middle_panel()` di row=5; `padx=16`

## 4. Bersihkan `_build_form_area()`

- [x] 4.1 `app.py` — hapus dari `_build_form_area()`: model toggle (row=3), error label (row=4), btn_frame (row=5), result_frame (row=6), calculator (row=7) — ini sudah dipindahkan ke `_build_middle_panel()`
- [x] 4.2 `app.py` — tambahkan heading kolom kiri `"Input Fitur"` di row=0 dalam left panel (sebelum dataset selector); jadikan dataset selector row=1, form container row=2

## 5. Verifikasi

- [ ] 5.1 Jalankan `python app.py` — konfirmasi window terbuka 1400×780 dengan 3 kolom terlihat
- [ ] 5.2 Pilih dataset berbeda — konfirmasi hanya form kiri yang berubah; tengah dan kanan tidak bergeser
- [ ] 5.3 Isi form dan klik Estimasi — konfirmasi hasil tampil di tengah, chart update di kanan
- [ ] 5.4 Scroll form pada COCOMO-81 (16 field) — konfirmasi kolom tengah dan kanan tidak ikut scroll
- [ ] 5.5 Klik Reset Form — konfirmasi semua field, result, dan kalkulator di tengah bersih

