import glob
import os

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_DIR = "data"
MODELS_DIR = "models"
CONFIG_FILE = "dataset_config.yaml"


def load_config() -> dict[str, dict]:
    if not os.path.exists(CONFIG_FILE):
        print(f"[WARNING] '{CONFIG_FILE}' tidak ditemukan. Jalankan auto_configure.py untuk membuat konfigurasi.")
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("datasets", {})


def get_config(stem: str) -> dict | None:
    return load_config().get(stem.lower())


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame, features: list[str], target: str):
    df = df[features + [target]].dropna()
    if len(df) < 5:
        raise ValueError(f"Dataset has only {len(df)} usable rows after dropping NaN — too few to train.")
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def train_models(X_train, y_train) -> tuple:
    svr = SVR(kernel="rbf")
    svr.fit(X_train, y_train)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return svr, lr


def evaluate(stem: str, models: dict, X_test, y_test) -> None:
    print(f"\n  [{stem}] --- Evaluation (Test Set) ---")
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"  [{stem}] {name:20s}  MAE={mae:.2f}  R2={r2:.4f}")


def train_and_export(csv_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
    config = get_config(stem)
    if config is None:
        print(f"[WARNING] No config found for '{stem}' in '{CONFIG_FILE}' — skipping.")
        print(f"          Run auto_configure.py to add a configuration for this file.")
        return False

    try:
        df = load_dataset(csv_path)
        X_train, X_test, y_train, y_test, scaler = preprocess(
            df, config["features"], config["target"]
        )
    except KeyError as exc:
        print(f"[WARNING] Column {exc} not found in '{os.path.basename(csv_path)}' — skipping.")
        print(f"          Check 'features' and 'target' in '{CONFIG_FILE}' for '{stem}'.")
        return False
    except ValueError as exc:
        print(f"[WARNING] {exc} — skipping '{stem}'.")
        return False

    svr, lr = train_models(X_train, y_train)
    evaluate(stem, {"SVR": svr, "LinearRegression": lr}, X_test, y_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(svr, os.path.join(MODELS_DIR, f"svr_{stem}.pkl"))
    joblib.dump(lr, os.path.join(MODELS_DIR, f"linreg_{stem}.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{stem}.pkl"))
    print(f"  [{stem}] Exported: svr_{stem}.pkl | linreg_{stem}.pkl | scaler_{stem}.pkl")
    return True


def cleanup_stale_models(trained_stems: set) -> int:
    """Delete .pkl trios in models/ whose stem is not in trained_stems. Returns count removed."""
    existing_svr = glob.glob(os.path.join(MODELS_DIR, "svr_*.pkl"))
    removed = 0
    for svr_path in existing_svr:
        stem = os.path.basename(svr_path)[4:-4]  # strip "svr_" prefix and ".pkl"
        if stem not in trained_stems:
            for prefix in ("svr_", "linreg_", "scaler_"):
                p = os.path.join(MODELS_DIR, f"{prefix}{stem}.pkl")
                if os.path.exists(p):
                    os.remove(p)
            print(f"  [CLEANUP] Removed stale model trio for '{stem}'.")
            removed += 1
    return removed


def run_batch() -> tuple[int, set]:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        print(f"[WARNING] No CSV files found in '{DATA_DIR}/'. Run downloader.py first.")
        return 0, set()

    print(f"[INFO] Found {len(csv_files)} CSV file(s) in '{DATA_DIR}/'.\n")
    succeeded = 0
    trained_stems: set = set()
    for path in csv_files:
        print(f"[INFO] Processing: {path}")
        if train_and_export(path):
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            trained_stems.add(stem)
            succeeded += 1

    return succeeded, trained_stems


if __name__ == "__main__":
    total = len(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    succeeded, trained_stems = run_batch()
    if total > 0:
        removed = cleanup_stale_models(trained_stems)
        print(f"\n[OK] Done. {succeeded}/{total} datasets trained successfully.")
        if removed:
            print(f"     Removed {removed} stale model trio(s) from '{MODELS_DIR}/'.")
        print(f"     Models saved to '{MODELS_DIR}/'.")
