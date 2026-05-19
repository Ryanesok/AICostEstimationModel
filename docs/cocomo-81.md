# Panduan Dataset COCOMO-81

## Overview

**COCOMO-81** (Constructive Cost Model) adalah dataset klasik dari riset Barry Boehm (1981) yang mencakup 63 proyek perangkat lunak nyata. Model ini menggunakan 15 **Cost Driver** (faktor pengali effort) ditambah ukuran kode (LOC) untuk memprediksi effort pengembangan.

| Atribut | Nilai |
|---|---|
| Jumlah proyek | 63 |
| Jumlah fitur input | 16 |
| Target | `actual` (effort aktual) |
| Satuan output | **Person-months** |
| Konversi | Tidak diperlukan — output langsung dalam person-months |

---

## Referensi Field

Semua Cost Driver menggunakan **skala rating** yang menentukan nilai pengali. Nilai yang lebih kecil atau lebih besar dari 1.00 masing-masing berarti pengurang atau penambah effort.

| Field | Label | Skala / Nilai Diterima | Contoh |
|---|---|---|---|
| `rely` | Reliability (Rely) | 0.75 · 0.88 · **1.00** · 1.15 · 1.40 | `1.00` |
| `data` | Database Size (Data) | 0.94 · **1.00** · 1.08 · 1.16 | `1.00` |
| `cplx` | Complexity (Cplx) | 0.70 · 0.85 · **1.00** · 1.15 · 1.30 · 1.65 | `1.00` |
| `time` | Execution Time Constraint (Time) | **1.00** · 1.11 · 1.30 · 1.66 | `1.00` |
| `stor` | Storage Constraint (Stor) | **1.00** · 1.06 · 1.21 · 1.56 | `1.00` |
| `virt` | Virtual Machine Volatility (Virt) | 0.87 · **1.00** · 1.15 · 1.30 | `1.00` |
| `turn` | Computer Turnaround (Turn) | 0.87 · **1.00** · 1.07 · 1.15 | `1.00` |
| `acap` | Analyst Capability (Acap) | 1.46 · 1.19 · **1.00** · 0.86 · 0.71 | `1.00` |
| `aexp` | Application Experience (Aexp) | 1.29 · 1.13 · **1.00** · 0.91 · 0.82 | `1.00` |
| `pcap` | Programmer Capability (Pcap) | 1.42 · 1.17 · **1.00** · 0.86 · 0.70 | `1.00` |
| `vexp` | VM Experience (Vexp) | 1.21 · 1.10 · **1.00** · 0.90 | `1.00` |
| `lexp` | Language Experience (Lexp) | 1.14 · 1.07 · **1.00** · 0.95 | `1.00` |
| `modp` | Modern Programming Practices (Modp) | 1.24 · 1.10 · **1.00** · 0.91 · 0.82 | `1.00` |
| `tool` | Software Tools (Tool) | 1.24 · 1.10 · **1.00** · 0.91 · 0.83 | `1.00` |
| `sced` | Schedule Constraint (Sced) | 1.23 · 1.08 · **1.00** · 1.04 · 1.10 | `1.00` |
| `loc` | LOC (ribuan baris kode) | Angka positif (ribuan) | `30` |

> **Nilai cetak tebal** = nilai nominal (kondisi rata-rata). Gunakan nilai ini sebagai titik awal jika tidak yakin.

---

## Penjelasan Tiap Cost Driver

### Faktor Produk
| Field | Sangat Rendah | Rendah | Nominal | Tinggi | Sangat Tinggi | Ekstra Tinggi |
|---|---|---|---|---|---|---|
| `rely` — Reliabilitas | 0.75 | 0.88 | 1.00 | 1.15 | 1.40 | — |
| `data` — Rasio DB/Program | 0.94 (D/P<10) | — | 1.00 (10–100) | 1.08 (100–1000) | 1.16 (≥1000) | — |
| `cplx` — Kompleksitas | 0.70 | 0.85 | 1.00 | 1.15 | 1.30 | 1.65 |

### Faktor Komputer
| Field | Sangat Rendah | Rendah | Nominal | Tinggi | Sangat Tinggi |
|---|---|---|---|---|---|
| `time` — Batasan waktu eksekusi | — | — | 1.00 (≤50%) | 1.11 (70%) | 1.30 (85%) |
| `stor` — Batasan memori | — | — | 1.00 (≤50%) | 1.06 (70%) | 1.21 (85%) |
| `virt` — Volatilitas VM | 0.87 (>1 bulan) | — | 1.00 (1 bulan) | 1.15 (1 minggu) | 1.30 (2 hari) |
| `turn` — Turnaround komputer | 0.87 (interaktif) | — | 1.00 (4 jam) | 1.07 (12 jam) | 1.15 (>1 hari) |

### Faktor Personel
| Field | Sangat Rendah | Rendah | Nominal | Tinggi | Sangat Tinggi |
|---|---|---|---|---|---|
| `acap` — Kemampuan analis | 1.46 | 1.19 | 1.00 | 0.86 | 0.71 |
| `aexp` — Pengalaman aplikasi | 1.29 (<4 bln) | 1.13 (1 thn) | 1.00 (3 thn) | 0.91 (6 thn) | 0.82 (>6 thn) |
| `pcap` — Kemampuan programmer | 1.42 | 1.17 | 1.00 | 0.86 | 0.70 |
| `vexp` — Pengalaman VM | 1.21 (<1 bln) | 1.10 (1 bln) | 1.00 (1 thn) | 0.90 (>3 thn) | — |
| `lexp` — Pengalaman bahasa | 1.14 (<1 bln) | 1.07 (1 bln) | 1.00 (1 thn) | 0.95 (>3 thn) | — |

### Faktor Proyek
| Field | Sangat Rendah | Rendah | Nominal | Tinggi | Sangat Tinggi |
|---|---|---|---|---|---|
| `modp` — Praktik modern | 1.24 | 1.10 | 1.00 | 0.91 | 0.82 |
| `tool` — Alat bantu | 1.24 | 1.10 | 1.00 | 0.91 | 0.83 |
| `sced` — Jadwal | 1.23 (dipercepat) | 1.08 | 1.00 (nominal) | 1.04 | 1.10 (diperlambat) |

### Ukuran
- **`loc`** — Jumlah kode dalam **ribuan baris** (KLOC). Proyek 30.000 baris = `30`.
