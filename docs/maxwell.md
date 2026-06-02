# Panduan Dataset Maxwell

## Overview

**Maxwell** adalah dataset dari riset Doris Maxwell (1996) yang mencakup 62 proyek perangkat lunak komersial yang disurvei antara 1985–1993. Dataset ini mencakup karakteristik organisasi dan 15 faktor praktik pengembangan (T-factors) untuk memprediksi effort.

| Atribut | Nilai |
|---|---|
| Jumlah proyek | 62 |
| Jumlah fitur input | 25 (setelah mengeluarkan `Duration`) |
| Target | `Effort` (person-hours) |
| Secondary target | `Duration` (tidak digunakan sebagai fitur) |
| Satuan output model | **Person-hours** |
| Tampilan aplikasi | **Person-months** (÷ 160 jam/bulan) |

> **Catatan konversi:** Model memprediksi dalam person-hours. Aplikasi membagi hasilnya dengan **160** sebelum ditampilkan sebagai person-months.

> **Catatan `Duration`:** Kolom `Duration` adalah output proyek (bukan input), sehingga dikecualikan dari fitur untuk menghindari kebocoran data.

---

## Referensi Field

### Field Konteks Proyek

| Field | Label | Satuan / Skala | Nilai Diterima | Contoh |
|---|---|---|---|---|
| `Syear` | Tahun Survey | tahun | 85–93 | `89` |
| `App` | Tipe Aplikasi | kategori | 1–5 (lihat bawah) | `1` |
| `Har` | Tipe Hardware | kategori | 1–5 (lihat bawah) | `2` |
| `Dba` | Database Administrator | kategori | 1–5 (lihat bawah) | `3` |
| `Ifc` | Kompleksitas Antarmuka | kategori | 1–5 (lihat bawah) | `3` |
| `Source` | Sumber Proyek | kategori | 1–5 (lihat bawah) | `1` |
| `Telonuse` | Penggunaan Sistem Online | kategori | 1–5 (lihat bawah) | `3` |
| `Nlan` | Jumlah Bahasa Pemrograman | unit | Angka positif (integer) | `2` |
| `Size` | Ukuran Sistem | Lines of Code | Angka positif | `15000` |
| `Time` | Waktu Pengembangan | jam | Angka positif | `1800` |

### T-Factors (Faktor Praktik Pengembangan)

Semua T-factor menggunakan skala yang sama: **1 = Tidak/Tidak ada** hingga **5 = Penuh/Komprehensif**.

| Field | Label | Deskripsi Singkat |
|---|---|---|
| `T01` | Faktor T01: Perangkat CASE | Penggunaan alat Computer-Aided Software Engineering |
| `T02` | Faktor T02: Prototyping | Penggunaan prototyping dalam pengembangan |
| `T03` | Faktor T03: Structured Programming | Penerapan pemrograman terstruktur |
| `T04` | Faktor T04: Code Review | Frekuensi review kode |
| `T05` | Faktor T05: Software Reuse | Tingkat reuse komponen perangkat lunak |
| `T06` | Faktor T06: Top-Down Design | Penerapan desain top-down |
| `T07` | Faktor T07: Chief Programmer Team | Penggunaan struktur tim chief programmer |
| `T08` | Faktor T08: Walkthroughs | Frekuensi walkthrough / inspeksi |
| `T09` | Faktor T09: Programming Standards | Penerapan standar pemrograman |
| `T10` | Faktor T10: Change Management | Formalitas manajemen perubahan |
| `T11` | Faktor T11: Configuration Management | Formalitas manajemen konfigurasi |
| `T12` | Faktor T12: Test Planning | Formalitas perencanaan pengujian |
| `T13` | Faktor T13: Structured Design | Penerapan desain terstruktur |
| `T14` | Faktor T14: Online Development | Tingkat pengembangan berbasis online/terminal |
| `T15` | Faktor T15: Project Management Tools | Penggunaan alat manajemen proyek |

---

## Skala Kategori

### `Syear` — Tahun Survey
Dua digit terakhir tahun: `85`=1985, `86`=1986, ..., `93`=1993.

### `App` — Tipe Aplikasi
| Nilai | Tipe |
|---|---|
| `1` | MIS (Management Information System) |
| `2` | Realtime / Embedded |
| `3` | Mathematical / Scientific |
| `4` | General purpose |
| `5` | Support / Infrastructure |

### `Har` — Tipe Hardware
| Nilai | Tipe |
|---|---|
| `1` | Mainframe |
| `2` | Minicomputer |
| `3` | Microcomputer |
| `4` | Workstation |
| `5` | PC |

### `Dba` — Peran Database Administrator
| Nilai | Peran |
|---|---|
| `1` | Tidak ada DBA |
| `2` | DBA jarang terlibat |
| `3` | DBA kadang terlibat |
| `4` | DBA sering terlibat |
| `5` | DBA penuh waktu |

### `Ifc` — Kompleksitas Antarmuka
| Nilai | Tingkat |
|---|---|
| `1` | Sangat rendah |
| `2` | Rendah |
| `3` | Sedang |
| `4` | Tinggi |
| `5` | Sangat tinggi |

### `Source` — Sumber Proyek
| Nilai | Sumber |
|---|---|
| `1` | Internal (in-house) |
| `2` | Vendor |
| `3` | Campuran internal + vendor |
| `4` | Outsource |
| `5` | Konsultan |

### `Telonuse` — Penggunaan Sistem Online
| Nilai | Tingkat |
|---|---|
| `1` | Tidak menggunakan |
| `2` | Minimal |
| `3` | Sebagian |
| `4` | Mayoritas |
| `5` | Penuh |

### T-Factors (T01–T15) — Skala Umum
| Nilai | Arti Umum |
|---|---|
| `1` | Tidak diterapkan |
| `2` | Diterapkan minimal / informal |
| `3` | Diterapkan sebagian / sedang |
| `4` | Diterapkan banyak / formal |
| `5` | Diterapkan penuh / komprehensif / ketat |

---

## Contoh Input Proyek Khas

| Field | Nilai Tipikal |
|---|---|
| `Syear` | `89` (1989) |
| `App` | `1` (MIS) |
| `Har` | `2` (Minicomputer) |
| `Dba` | `3` (Kadang terlibat) |
| `Ifc` | `3` (Sedang) |
| `Source` | `1` (Internal) |
| `Telonuse` | `3` (Sebagian) |
| `Nlan` | `2` |
| `T01`–`T15` | `3` (semua sedang) |
| `Size` | `15000` |
| `Time` | `1800` |
