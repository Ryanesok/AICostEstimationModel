import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_DIR = "data"
MODELS_DIR = "models"
CONFIG_FILE = "dataset_config.yaml"

GRID_PARAMS = {
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 1.0],
    "kernel": ["rbf", "linear"],
}


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
    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} usable rows after dropping NaN — "
            "cross-validation membutuhkan minimal 10 baris."
        )
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def evaluate_models(X: np.ndarray, y: np.ndarray) -> dict:
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": "r2",
    }

    gs = GridSearchCV(
        SVR(), GRID_PARAMS, cv=kfold,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    gs.fit(X, y)

    svr_cv = cross_validate(gs.best_estimator_, X, y, cv=kfold, scoring=scoring)
    lr_cv = cross_validate(LinearRegression(), X, y, cv=kfold, scoring=scoring)

    return {
        "best_svr_params": gs.best_params_,
        "cv_mae_mean": float(-svr_cv["test_mae"].mean()),
        "cv_mae_std": float(svr_cv["test_mae"].std()),
        "cv_r2_mean": float(svr_cv["test_r2"].mean()),
        "cv_r2_std": float(svr_cv["test_r2"].std()),
        "lr_cv_mae_mean": float(-lr_cv["test_mae"].mean()),
        "lr_cv_r2_mean": float(lr_cv["test_r2"].mean()),
    }


def save_metrics(stem: str, metrics: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{stem}_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def train_and_export(csv_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
    config = get_config(stem)
    if config is None:
        print(f"[WARNING] No config found for '{stem}' in '{CONFIG_FILE}' — skipping.")
        print(f"          Run auto_configure.py to add a configuration for this file.")
        return False

    try:
        df = load_dataset(csv_path)
    except Exception as exc:
        print(f"[WARNING] Gagal membaca '{os.path.basename(csv_path)}': {exc} — skipping.")
        return False

    if config["target"] not in df.columns:
        print(f"[WARNING] Primary target column '{config['target']}' tidak ditemukan di '{os.path.basename(csv_path)}' — skipping.")
        print(f"          Periksa 'target' di '{CONFIG_FILE}' untuk '{stem}'.")
        return False

    # Exclude secondary_target from features at runtime (guard against stale configs)
    secondary = config.get("secondary_target")
    features = [f for f in config["features"] if f != secondary]

    try:
        X, y, scaler = preprocess(df, features, config["target"])
    except KeyError as exc:
        print(f"[WARNING] Column {exc} not found in '{os.path.basename(csv_path)}' — skipping.")
        print(f"          Check 'features' and 'target' in '{CONFIG_FILE}' for '{stem}'.")
        return False
    except ValueError as exc:
        print(f"[WARNING] {exc} — skipping '{stem}'.")
        return False

    print(f"  [{stem}] Menjalankan 5-fold CV + GridSearchCV ({len(y)} baris) ...")
    metrics = evaluate_models(X, y)
    metrics["trained_on_rows"] = int(len(y))

    print(f"  [{stem}] SVR  CV-MAE={metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f}  CV-R²={metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print(f"  [{stem}] LR   CV-MAE={metrics['lr_cv_mae_mean']:.2f}  CV-R²={metrics['lr_cv_r2_mean']:.4f}")
    print(f"  [{stem}] Best SVR params: {metrics['best_svr_params']}")

    final_svr = SVR(**metrics["best_svr_params"])
    final_svr.fit(X, y)
    final_lr = LinearRegression()
    final_lr.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_svr, os.path.join(MODELS_DIR, f"svr_{stem}.pkl"))
    joblib.dump(final_lr, os.path.join(MODELS_DIR, f"linreg_{stem}.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{stem}.pkl"))
    save_metrics(stem, metrics)
    print(f"  [{stem}] Exported: svr_{stem}.pkl | linreg_{stem}.pkl | scaler_{stem}.pkl | {stem}_metrics.json")
    return True


def cleanup_stale_models(trained_stems: set) -> int:
    existing_svr = glob.glob(os.path.join(MODELS_DIR, "svr_*.pkl"))
    removed = 0
    for svr_path in existing_svr:
        stem = os.path.basename(svr_path)[4:-4]
        if stem not in trained_stems:
            for prefix in ("svr_", "linreg_", "scaler_"):
                p = os.path.join(MODELS_DIR, f"{prefix}{stem}.pkl")
                if os.path.exists(p):
                    os.remove(p)
            metrics_path = os.path.join(MODELS_DIR, f"{stem}_metrics.json")
            if os.path.exists(metrics_path):
                os.remove(metrics_path)
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
