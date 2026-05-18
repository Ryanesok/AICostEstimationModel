"""
auto_configure.py - Konfigurasi otomatis dataset baru untuk model_pipeline.py

Cara pakai:
    python auto_configure.py

Program ini akan:
  1. Scan semua file CSV di folder data/
  2. Lewati dataset yang sudah ada di dataset_config.yaml
  3. Untuk dataset baru: tampilkan daftar kolom numerik dan minta user memilih kolom target
  4. Simpan konfigurasi baru ke dataset_config.yaml
"""

import glob
import os
import sys

import pandas as pd
import yaml

DATA_DIR = "data"
CONFIG_FILE = "dataset_config.yaml"

# Kolom yang umumnya bukan fitur (metadata/identifier)
EXCLUDE_COLS = {
    "id", "project", "yearend", "year", "language",
    "pointsnonadjust", "pointsadjust", "pointsajust", "adjustment",
    "no", "num", "index", "idx",
}

# Nama kolom yang kemungkinan besar adalah target (effort)
EFFORT_COLS = {
    "effort", "actualeffortmonths", "cost", "duration",
    "manmonths", "personmonths", "actualeffort",
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("datasets", {})


def save_config(datasets: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(
            {"datasets": datasets},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f"\n[OK] Konfigurasi disimpan ke '{CONFIG_FILE}'.")


def is_exclude_col(col: str) -> bool:
    return col.lower() in EXCLUDE_COLS


def is_effort_col(col: str) -> bool:
    return col.lower() in EFFORT_COLS


def configure_one(csv_path: str, existing: dict) -> tuple[str, dict] | None:
    stem = os.path.splitext(os.path.basename(csv_path))[0].lower()

    if stem in existing:
        print(f"  [SKIP] '{stem}' sudah ada dalam konfigurasi.")
        return None

    print(f"\n{'='*50}")
    print(f"  Dataset baru: {stem}  ({csv_path})")
    print(f"{'='*50}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  [ERROR] Gagal membaca CSV: {exc}")
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("  [WARN] Tidak ada kolom numerik — dilewati.")
        return None

    # Filter obvious metadata columns
    candidates = [c for c in numeric_cols if not is_exclude_col(c)]
    if not candidates:
        print("  [WARN] Semua kolom numerik terdeteksi sebagai metadata — dilewati.")
        return None

    # Suggest a target column
    suggested = next((c for c in candidates if is_effort_col(c)), None)

    print(f"\n  Kolom numerik yang tersedia ({len(candidates)} kolom):")
    for i, col in enumerate(candidates, 1):
        tag = "  <-- disarankan sebagai target" if col == suggested else ""
        print(f"    {i:2d}. {col}{tag}")

    print()
    prompt = f"  Pilih kolom TARGET (output prediksi) [1-{len(candidates)}] atau ketik nama kolom: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("  Input kosong. Coba lagi.")
            continue
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                target = candidates[idx]
                break
            print(f"  Angka di luar rentang. Masukkan 1-{len(candidates)}.")
        except ValueError:
            if raw in candidates:
                target = raw
                break
            print(f"  Kolom '{raw}' tidak ditemukan. Coba lagi.")

    features = [c for c in candidates if c != target]

    if not features:
        print("  [ERROR] Tidak ada kolom fitur tersisa setelah memilih target.")
        return None

    print(f"\n  Target   : {target}")
    print(f"  Fitur    : {', '.join(features)}")
    print(f"  Jumlah   : {len(features)} fitur")

    confirm = input("\n  Simpan konfigurasi ini? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Dibatalkan.")
        return None

    labels, hints = collect_labels_hints(features)

    entry: dict = {"features": features, "target": target}
    if labels:
        entry["labels"] = labels
    if hints:
        entry["hints"] = hints
    return stem, entry


def collect_labels_hints(features: list[str]) -> tuple[dict, dict]:
    """Interactively collect friendly labels and hints for each feature column."""
    print("\n  Isi label tampilan dan hint untuk setiap fitur.")
    print("  Tekan Enter untuk melewati (akan menggunakan nama kolom sebagai label).\n")
    add = input("  Tambahkan label/hint sekarang? [y/N]: ").strip().lower()
    if add != "y":
        return {}, {}

    labels: dict = {}
    hints: dict = {}
    for col in features:
        label = input(f"    Label '{col}' [Enter = '{col}']: ").strip()
        if label:
            labels[col] = label
        hint = input(f"    Hint  '{col}' [Enter = kosong]: ").strip()
        if hint:
            hints[col] = hint
    return labels, hints


def cleanup_stale_configs(existing: dict, csv_files: list[str]) -> dict:
    """Remove config entries that have no matching CSV in data/. Returns updated dict."""
    csv_stems = {os.path.splitext(os.path.basename(f))[0].lower() for f in csv_files}
    stale = [stem for stem in existing if stem not in csv_stems]
    if not stale:
        return existing

    print(f"\n  Konfigurasi tanpa CSV yang sesuai: {', '.join(stale)}")
    confirm = input("  Hapus konfigurasi usang ini? [y/N]: ").strip().lower()
    if confirm == "y":
        for stem in stale:
            del existing[stem]
            print(f"  [REMOVED] Konfigurasi '{stem}' dihapus.")
    return existing


def main() -> None:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    if not csv_files:
        print(f"[WARNING] Tidak ada file CSV di '{DATA_DIR}/'. Letakkan CSV di sana lalu jalankan ulang.")
        sys.exit(0)

    print(f"[INFO] Ditemukan {len(csv_files)} file CSV di '{DATA_DIR}/'.")

    existing = load_config()
    original_keys = set(existing.keys())
    if existing:
        print(f"[INFO] Konfigurasi yang sudah ada: {', '.join(existing.keys())}")

    # Remove configs for CSVs that no longer exist
    existing = cleanup_stale_configs(existing, csv_files)

    new_count = 0
    for path in csv_files:
        result = configure_one(path, existing)
        if result is not None:
            stem, cfg = result
            existing[stem] = cfg
            new_count += 1

    changed = new_count > 0 or set(existing.keys()) != original_keys
    if changed:
        save_config(existing)
        if new_count > 0:
            print(f"\n[OK] {new_count} dataset baru dikonfigurasi.")
            print(f"     Jalankan 'python model_pipeline.py' untuk melatih model.")
    else:
        print("\n[INFO] Tidak ada perubahan pada konfigurasi.")


if __name__ == "__main__":
    main()
