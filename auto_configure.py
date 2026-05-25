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
FIELD_LABELS_FILE = "field_labels.yaml"

# Kolom yang umumnya bukan fitur (metadata/identifier)
EXCLUDE_COLS = {
    "id", "project", "yearend", "year", "language",
    "pointsnonadjust", "pointsadjust", "pointsajust", "adjustment",
    "no", "num", "index", "idx",
    # kitchenham identifiers
    "client.code",
    # nasa93 identifiers
    "recordnumber", "center",
}

# Primary target keywords — direct effort/cost metrics; checked first in suggest_target()
PRIMARY_TARGET_COLS = {
    "effort", "actualeffort", "actualeffortmonths", "cost",
    "manmonths", "personmonths", "workhours", "totalhours", "totaleffort",
    "devtime", "developmenttime", "projectduration", "actual"
}

# Secondary target keywords — time/duration metrics; checked only if no primary match
SECONDARY_TARGET_COLS = {
    "duration", "time", "elapsed",
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


def is_primary_target_col(col: str) -> bool:
    return col.lower() in PRIMARY_TARGET_COLS


def is_secondary_target_col(col: str) -> bool:
    return col.lower() in SECONDARY_TARGET_COLS


def load_default_labels(stem: str | None = None) -> dict:
    """Load field_labels.yaml. If stem given, return {labels, hints, effort_unit, ...} for that stem."""
    if not os.path.exists(FIELD_LABELS_FILE):
        return {}
    with open(FIELD_LABELS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    datasets = data.get("datasets", {})
    if stem is not None:
        return datasets.get(stem.lower(), {})
    return datasets


def load_post_project_cols(stem: str) -> list[str]:
    """Return known post-project columns for stem from field_labels.yaml."""
    defaults = load_default_labels(stem)
    return defaults.get("post_project_cols", [])


def backfill_defaults(existing: dict) -> tuple[dict, int]:
    """Apply field_labels.yaml defaults to existing config entries missing labels/hints/effort_unit/secondary_target."""
    all_defaults = load_default_labels()
    if not all_defaults:
        return existing, 0
    updated = 0
    for stem, cfg in existing.items():
        defaults = all_defaults.get(stem, {})
        if not defaults:
            continue
        changed = False
        for key in ("labels", "hints", "effort_unit"):
            if not cfg.get(key) and defaults.get(key):
                cfg[key] = defaults[key]
                changed = True
        # secondary_target: backfill and remove from features if still there
        if not cfg.get("secondary_target") and defaults.get("secondary_target"):
            sec = defaults["secondary_target"]
            cfg["secondary_target"] = sec
            changed = True
            if sec in cfg.get("features", []):
                cfg["features"] = [f for f in cfg["features"] if f != sec]
                print(f"  [BACKFILL] '{sec}' dipindahkan dari fitur ke secondary_target untuk '{stem}'.")
        # post_project_cols: remove any that still appear in features
        for col in defaults.get("post_project_cols", []):
            if col in cfg.get("features", []):
                cfg["features"] = [f for f in cfg["features"] if f != col]
                print(f"  [BACKFILL] '{col}' dihapus dari fitur '{stem}' — nilai post-project.")
                changed = True
        if changed:
            print(f"  [BACKFILL] Default label/hint/effort_unit diterapkan untuk '{stem}'.")
            updated += 1
    return existing, updated


def suggest_target(df: pd.DataFrame, candidates: list[str]) -> tuple[str | None, str]:
    """Return (suggested_column, reason) for the most likely target column.

    Priority:
    1. Name matches a PRIMARY_TARGET_COLS keyword (direct effort/cost metric).
    2. Name matches a SECONDARY_TARGET_COLS keyword (time/duration metric).
    3. Column with highest mean absolute correlation to the others.
    """
    for c in candidates:
        if is_primary_target_col(c):
            return c, "primary keyword"
    for c in candidates:
        if is_secondary_target_col(c):
            return c, "secondary keyword"
    if len(candidates) < 2:
        return (candidates[0] if candidates else None), "satu-satunya kandidat"
    corr = df[candidates].corr().abs()
    mean_corr = (corr.sum() - 1.0) / (len(candidates) - 1)
    return str(mean_corr.idxmax()), "korelasi tertinggi dengan kolom lain"


def find_near_constant(df: pd.DataFrame, candidates: list[str], threshold: float = 0.95) -> list[str]:
    """Return columns where one value covers ≥ threshold of non-null rows."""
    near_constant = []
    for col in candidates:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        top_ratio = series.value_counts().iloc[0] / len(series)
        if top_ratio >= threshold:
            near_constant.append(col)
    return near_constant


def check_leakage(
    df: pd.DataFrame, features: list[str], target: str, threshold: float = 0.95
) -> list[tuple[str, float]]:
    """Return (feature, r) pairs where |Pearson r| with target ≥ threshold."""
    target_series = df[target].dropna()
    suspicious: list[tuple[str, float]] = []
    for col in features:
        feat_series = df[col].dropna()
        common = feat_series.index.intersection(target_series.index)
        if len(common) < 3:
            continue
        r = float(feat_series[common].corr(target_series[common]))
        if abs(r) >= threshold:
            suspicious.append((col, r))
    return suspicious


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

    # Filter metadata columns
    candidates = [c for c in numeric_cols if not is_exclude_col(c)]
    if not candidates:
        print("  [WARN] Semua kolom numerik terdeteksi sebagai metadata — dilewati.")
        return None

    # Exclude known post-project columns before showing candidates
    post_project = load_post_project_cols(stem)
    excluded_pp = [c for c in candidates if c in post_project]
    if excluded_pp:
        for col in excluded_pp:
            candidates.remove(col)
            print(f"  [FILTER] Kolom '{col}' dikecualikan — nilai post-project (hanya diketahui setelah proyek selesai).")
        if not candidates:
            print("  [ERROR] Tidak ada kolom fitur tersisa setelah filter post-project.")
            return None

    # Drop near-constant columns before showing candidates
    near_constant = find_near_constant(df, candidates)
    if near_constant:
        for col in near_constant:
            candidates.remove(col)
            print(f"  [FILTER] Kolom '{col}' dihapus — nilai dominan mencakup ≥95% baris (near-constant).")
        if not candidates:
            print("  [ERROR] Tidak ada kolom fitur tersisa setelah filter near-constant.")
            return None

    suggested, suggested_reason = suggest_target(df, candidates)

    print(f"\n  Kolom numerik yang tersedia ({len(candidates)} kolom):")
    for i, col in enumerate(candidates, 1):
        tag = f"  <-- disarankan sebagai target ({suggested_reason})" if col == suggested else ""
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

    # Leakage check
    suspicious = check_leakage(df, features, target)
    if suspicious:
        print(f"\n  [LEAKAGE WARNING] Fitur berikut berkorelasi sangat tinggi dengan target '{target}':")
        for col, r in suspicious:
            print(f"    - '{col}'  |r| = {abs(r):.4f}  (kemungkinan data leakage)")
        print()
        drop_list: list[str] = []
        for col, r in suspicious:
            ans = input(f"  Hapus '{col}' dari fitur? [y/N]: ").strip().lower()
            if ans == "y":
                drop_list.append(col)
                print(f"  [REMOVED] '{col}' dihapus dari fitur.")
        features = [f for f in features if f not in drop_list]

        if not features:
            print("  [ERROR] Tidak ada fitur tersisa setelah leakage check — konfigurasi dibatalkan.")
            return None

    confirm = input("\n  Simpan konfigurasi ini? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Dibatalkan.")
        return None

    labels, hints = collect_labels_hints(features, stem=stem)

    entry: dict = {"features": features, "target": target}
    if labels:
        entry["labels"] = labels
    if hints:
        entry["hints"] = hints
    return stem, entry


def collect_labels_hints(features: list[str], stem: str = "") -> tuple[dict, dict]:
    """Interactively collect friendly labels and hints for each feature column.

    Falls back to field_labels.yaml defaults if user skips. When entering manually,
    defaults are shown as placeholder values in each prompt.
    """
    defaults = load_default_labels(stem) if stem else {}
    default_labels = defaults.get("labels", {})
    default_hints = defaults.get("hints", {})
    has_defaults = bool(default_labels or default_hints)

    print("\n  Isi label tampilan dan hint untuk setiap fitur.")
    if has_defaults:
        print(f"  (default tersedia dari '{FIELD_LABELS_FILE}' — akan digunakan otomatis jika dilewati)")
    print("  Tekan Enter untuk melewati.\n")

    add = input("  Tambahkan label/hint secara manual? [y/N]: ").strip().lower()
    if add != "y":
        if has_defaults:
            print(f"  [AUTO] Menggunakan label/hint default dari '{FIELD_LABELS_FILE}'.")
        return dict(default_labels), dict(default_hints)

    labels: dict = {}
    hints: dict = {}
    for col in features:
        dl = default_labels.get(col, col)
        dh = default_hints.get(col, "")
        dh_preview = (dh[:40] + "...") if len(dh) > 40 else dh

        label = input(f"    Label '{col}' [Enter = '{dl}']: ").strip()
        labels[col] = label if label else dl

        hint = input(f"    Hint  '{col}' [Enter = '{dh_preview}']: ").strip()
        hints[col] = hint if hint else dh

    # Drop entries that are just the bare column name (no added value)
    labels = {k: v for k, v in labels.items() if v and v != k}
    hints = {k: v for k, v in hints.items() if v}
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

    existing, backfilled = backfill_defaults(existing)

    changed = new_count > 0 or backfilled > 0 or set(existing.keys()) != original_keys
    if changed:
        save_config(existing)
        if new_count > 0:
            print(f"\n[OK] {new_count} dataset baru dikonfigurasi.")
            print(f"     Jalankan 'python model_pipeline.py' untuk melatih model.")
        if backfilled > 0:
            print(f"[OK] {backfilled} dataset diperbarui dengan label/hint/effort_unit default.")
    else:
        print("\n[INFO] Tidak ada perubahan pada konfigurasi.")


if __name__ == "__main__":
    main()
