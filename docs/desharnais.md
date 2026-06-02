# Panduan Dataset Desharnais

## Overview

**Desharnais** adalah dataset dari riset J.-M. Desharnais (1988) yang mencakup 81 proyek sistem informasi dari satu perusahaan Kanada. Dataset ini menggunakan ukuran fungsional (Function Points) dan karakteristik tim untuk memprediksi effort pengembangan.

| Atribut | Nilai |
|---|---|
| Jumlah proyek | 81 |
| Jumlah fitur input | 5 |
| Target | `Effort` (person-hours) |
| Satuan output model | **Person-hours** |
| Tampilan aplikasi | **Person-months** (÷ 160 jam/bulan) |

> **Catatan konversi:** Model memprediksi dalam person-hours. Aplikasi membagi hasilnya dengan **160** sebelum ditampilkan sebagai person-months.

---

## Referensi Field

| Field | Label | Satuan | Nilai Diterima | Contoh |
|---|---|---|---|---|
| `TeamExp` | Team Experience | tahun | Angka positif | `3` |
| `ManagerExp` | Manager Experience | tahun | Angka positif | `5` |
| `Length` | Durasi Proyek | bulan | Angka positif | `12` |
| `Transactions` | Jumlah Transaksi | unit | Angka positif | `45` |
| `Entities` | Jumlah Entitas Data | unit | Angka positif | `20` |

---

## Penjelasan Field

| Field | Deskripsi |
|---|---|
| `TeamExp` | Pengalaman rata-rata anggota tim pengembang dalam tahun. Tim berpengalaman cenderung menghasilkan effort yang lebih efisien. |
| `ManagerExp` | Pengalaman manajer proyek dalam tahun. Manajer berpengalaman lebih baik dalam perencanaan dan pengendalian proyek. |
| `Length` | Durasi total proyek dalam bulan dari awal hingga selesai. |
| `Transactions` | Jumlah transaksi fungsional yang diproses oleh sistem (komponen Function Points). |
| `Entities` | Jumlah entitas data dalam model data sistem (komponen Function Points). |

---

## Contoh Input Proyek Khas

| Field | Proyek Kecil | Proyek Sedang | Proyek Besar |
|---|---|---|---|
| `TeamExp` | 1 | 3 | 6 |
| `ManagerExp` | 2 | 5 | 8 |
| `Length` | 6 | 12 | 24 |
| `Transactions` | 15 | 45 | 120 |
| `Entities` | 8 | 20 | 50 |
