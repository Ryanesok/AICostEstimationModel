# Panduan Dataset China

## Overview

**China** adalah dataset yang dikumpulkan dari proyek-proyek perangkat lunak di Cina (499 proyek) yang dipublikasikan dalam konteks riset ISBSG. Dataset ini menggunakan **Function Points** sebagai ukuran fungsionalitas sistem untuk memprediksi effort pengembangan.

| Atribut | Nilai |
|---|---|
| Jumlah proyek | 499 |
| Jumlah fitur input | 14 (setelah mengeluarkan `Duration`) |
| Target | `Effort` (person-hours) |
| Secondary target | `Duration` (tidak digunakan sebagai fitur) |
| Satuan output model | **Person-hours** |
| Tampilan aplikasi | **Person-months** (÷ 160 jam/bulan) |

> **Catatan konversi:** Model memprediksi dalam person-hours. Aplikasi membagi hasilnya dengan **160** sebelum ditampilkan sebagai person-months.

> **Catatan `Duration`:** Kolom `Duration` adalah output proyek (bukan input), sehingga dikecualikan dari fitur untuk menghindari kebocoran data.

---

## Referensi Field

| Field | Label | Satuan / Skala | Nilai Diterima | Contoh |
|---|---|---|---|---|
| `AFP` | Adjusted Function Points | FP | Angka positif | `215` |
| `Input` | External Inputs (EI) | unit | Angka positif (integer) | `10` |
| `Output` | External Outputs (EO) | unit | Angka positif (integer) | `8` |
| `Enquiry` | External Enquiries (EQ) | unit | Angka positif (integer) | `12` |
| `File` | Internal Logical Files (ILF) | unit | Angka positif (integer) | `5` |
| `Interface` | External Interface Files (EIF) | unit | Angka positif (integer) | `3` |
| `Added` | FP Ditambahkan | FP | Angka ≥ 0 | `50` |
| `Changed` | FP Dimodifikasi | FP | Angka ≥ 0 | `30` |
| `Deleted` | FP Dihapus | FP | Angka ≥ 0 | `10` |
| `PDR_AFP` | PDR/AFP Ratio | rasio | Angka positif (desimal) | `1.2` |
| `PDR_UFP` | PDR/UFP Ratio | rasio | Angka positif (desimal) | `0.9` |
| `NPDR_AFP` | NPDR/AFP Ratio | rasio | Angka positif (desimal) | `0.8` |
| `NPDU_UFP` | NPDU/UFP Ratio | rasio | Angka positif (desimal) | `0.7` |
| `Resource` | Jumlah Resource/Tim | kategori | 1 · 2 · 3 · 4 | `2` |

---

## Penjelasan Field

### Function Point Components

| Field | Deskripsi |
|---|---|
| `AFP` | **Adjusted Function Points** — total FP setelah diterapkan Value Adjustment Factor (VAF). Ini adalah ukuran utama ukuran fungsional sistem. |
| `Input` | **External Inputs** — jumlah proses yang menerima data dari luar batas sistem (form input, batch load). |
| `Output` | **External Outputs** — jumlah proses yang mengirimkan data ke luar batas sistem (laporan, file ekspor). |
| `Enquiry` | **External Enquiries** — jumlah proses query yang menampilkan data tanpa transformasi berarti (lookup, search). |
| `File` | **Internal Logical Files (ILF)** — jumlah grup data yang dikelola dan diperbarui oleh sistem itu sendiri. |
| `Interface` | **External Interface Files (EIF)** — jumlah grup data yang digunakan sistem tapi dikelola sistem lain. |

### Enhancement Fields (untuk proyek perbaikan)

| Field | Deskripsi |
|---|---|
| `Added` | FP yang ditambahkan pada proyek enhancement. Bernilai 0 untuk proyek baru. |
| `Changed` | FP yang dimodifikasi pada proyek enhancement. Bernilai 0 untuk proyek baru. |
| `Deleted` | FP yang dihapus pada proyek enhancement. Bernilai 0 untuk proyek baru. |

### Ratio Fields

| Field | Deskripsi |
|---|---|
| `PDR_AFP` | Rasio *Preliminary Design Review* terhadap AFP. Mengukur proporsi review awal. |
| `PDR_UFP` | Rasio *Preliminary Design Review* terhadap *Unadjusted FP*. |
| `NPDR_AFP` | Rasio Non-PDR terhadap AFP. |
| `NPDU_UFP` | Rasio Non-PDR terhadap Unadjusted FP. |

### Resource

| Nilai | Arti |
|---|---|
| `1` | Tim kecil |
| `2` | Tim sedang |
| `3` | Tim besar |
| `4` | Tim sangat besar |

---

## Contoh Input Proyek Khas

| Field | Proyek Kecil | Proyek Sedang | Proyek Besar |
|---|---|---|---|
| `AFP` | 80 | 215 | 600 |
| `Input` | 5 | 10 | 30 |
| `Output` | 4 | 8 | 20 |
| `Enquiry` | 6 | 12 | 35 |
| `File` | 3 | 5 | 15 |
| `Interface` | 1 | 3 | 8 |
| `Added` | 0 | 50 | 150 |
| `Changed` | 0 | 30 | 80 |
| `Deleted` | 0 | 10 | 25 |
| `PDR_AFP` | 1.0 | 1.2 | 1.5 |
| `PDR_UFP` | 0.8 | 0.9 | 1.1 |
| `NPDR_AFP` | 0.7 | 0.8 | 1.0 |
| `NPDU_UFP` | 0.6 | 0.7 | 0.9 |
| `Resource` | 1 | 2 | 3 |
