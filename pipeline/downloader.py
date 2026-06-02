import os
from pathlib import Path
from urllib.parse import urlparse

import requests

_PIPELINE_DIR = Path(__file__).parent
_ROOT_DIR = _PIPELINE_DIR.parent

DATA_DIR = str(_ROOT_DIR / "data")
DATASETS_FILE = str(_PIPELINE_DIR / "datasets.txt")
REQUEST_TIMEOUT = 30


def parse_urls(filepath: str) -> list[str]:
    urls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def arff_to_csv(content_bytes: bytes, dest_path: str) -> bool:
    """Parse ARFF binary content and write as CSV to dest_path. Returns True on success."""
    try:
        text = content_bytes.decode("utf-8", errors="replace")
    except Exception:
        return False

    attributes: list[str] = []
    data_rows: list[str] = []
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        upper = line.upper()
        if upper.startswith("@ATTRIBUTE"):
            parts = line.split(None, 2)
            if len(parts) >= 2:
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)
        elif upper.startswith("@DATA"):
            in_data = True
        elif in_data:
            # Replace ARFF missing-value marker '?' with empty string
            data_rows.append(line.replace("?", ""))

    if not attributes or not data_rows:
        return False

    with open(dest_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(attributes) + "\n")
        for row in data_rows:
            f.write(row + "\n")

    return True


def download_file(url: str, dest_dir: str) -> str:
    """Download one URL. Returns 'downloaded', 'skipped', or 'failed'."""
    filename = filename_from_url(url)
    if not filename:
        print(f"  [WARN] Cannot determine filename from: {url}")
        return "failed"

    is_arff = filename.lower().endswith(".arff")
    out_filename = filename[:-5] + ".csv" if is_arff else filename
    dest_path = os.path.join(dest_dir, out_filename)

    if os.path.exists(dest_path):
        print(f"  [SKIP] '{out_filename}' sudah ada — dilewati.")
        return "skipped"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        if is_arff:
            if not arff_to_csv(response.content, dest_path):
                print(f"  [WARN] Gagal mengonversi ARFF: '{filename}' — format tidak dikenali.")
                return "failed"
            size_kb = len(response.content) / 1024
            print(f"  [OK]   '{filename}' diunduh dan dikonversi ke '{out_filename}' ({size_kb:.1f} KB).")
        else:
            with open(dest_path, "wb") as f:
                f.write(response.content)
            size_kb = len(response.content) / 1024
            print(f"  [OK]   '{out_filename}' diunduh ({size_kb:.1f} KB).")

        return "downloaded"
    except Exception as exc:
        print(f"  [WARN] Gagal mengunduh: {url}")
        print(f"         Alasan: {exc}")
        return "failed"


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATASETS_FILE):
        print(f"[ERROR] '{DATASETS_FILE}' tidak ditemukan.")
        raise SystemExit(1)

    urls = parse_urls(DATASETS_FILE)
    if not urls:
        print("[INFO] Tidak ada URL di datasets.txt.")
        raise SystemExit(0)

    print(f"[INFO] Memproses {len(urls)} URL...\n")
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for url in urls:
        result = download_file(url, DATA_DIR)
        counts[result] += 1

    print(
        f"\n[Ringkasan] Diunduh: {counts['downloaded']} | "
        f"Di-skip: {counts['skipped']} | "
        f"Gagal: {counts['failed']}"
    )
