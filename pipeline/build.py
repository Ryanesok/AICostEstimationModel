"""
pipeline/build.py — Konfigurasi dan pelatihan model dalam satu modul.

Cara pakai:
    python pipeline/build.py              # configure + train (build_all)
    python pipeline/build.py --configure  # hanya konfigurasi dataset baru
    python pipeline/build.py --train      # hanya melatih model

Fungsi publik:
    configure()   — scan CSV baru dan konfigurasi interaktif
    train()       — latih semua model berdasarkan konfigurasi yang ada
    build_all()   — configure() lalu train()
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

_PIPELINE_DIR = Path(__file__).parent
_ROOT_DIR = _PIPELINE_DIR.parent

DATA_DIR = str(_ROOT_DIR / "data")
MODELS_DIR = str(_ROOT_DIR / "models")
CONFIG_FILE = str(_PIPELINE_DIR / "dataset_config.yaml")
FIELD_LABELS_FILE = str(_PIPELINE_DIR / "field_labels.yaml")

# ── auto_configure constants ───────────────────────────────────────────────────

EXCLUDE_COLS = {
    "id", "project", "yearend", "year", "language",
    "pointsnonadjust", "pointsadjust", "pointsajust", "adjustment",
    "no", "num", "index", "idx",
    "client.code",
    "recordnumber", "center",
}

PRIMARY_TARGET_COLS = {
    "effort", "actualeffort", "actualeffortmonths", "cost",
    "manmonths", "personmonths", "workhours", "totalhours", "totaleffort",
    "devtime", "developmenttime", "projectduration", "actual"
}

SECONDARY_TARGET_COLS = {
    "duration", "time", "elapsed",
}

# ── model_pipeline grid params ─────────────────────────────────────────────────

GRID_PARAMS = {
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 1.0],
    "kernel": ["rbf", "linear"],
}

XGB_GRID_PARAMS = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}

RF_GRID_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

LSTM_GRID_PARAMS = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "lr": [1e-3, 5e-4],
}


# ── Model definition ───────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


# ── Shared config helpers ──────────────────────────────────────────────────────

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        print(f"[WARNING] '{CONFIG_FILE}' tidak ditemukan. Jalankan 'python pipeline/build.py --configure' untuk membuat konfigurasi.")
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


def get_config(stem: str) -> dict | None:
    return load_config().get(stem.lower())


# ── auto_configure helpers ─────────────────────────────────────────────────────

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
        if not cfg.get("secondary_target") and defaults.get("secondary_target"):
            sec = defaults["secondary_target"]
            cfg["secondary_target"] = sec
            changed = True
            if sec in cfg.get("features", []):
                cfg["features"] = [f for f in cfg["features"] if f != sec]
                print(f"  [BACKFILL] '{sec}' dipindahkan dari fitur ke secondary_target untuk '{stem}'.")
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

    candidates = [c for c in numeric_cols if not is_exclude_col(c)]
    if not candidates:
        print("  [WARN] Semua kolom numerik terdeteksi sebagai metadata — dilewati.")
        return None

    post_project = load_post_project_cols(stem)
    excluded_pp = [c for c in candidates if c in post_project]
    if excluded_pp:
        for col in excluded_pp:
            candidates.remove(col)
            print(f"  [FILTER] Kolom '{col}' dikecualikan — nilai post-project (hanya diketahui setelah proyek selesai).")
        if not candidates:
            print("  [ERROR] Tidak ada kolom fitur tersisa setelah filter post-project.")
            return None

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

    try:
        target_skew = float(df[target].dropna().skew())
        log_transform = abs(target_skew) > 1.0
        print(f"  [SKEWNESS] target '{target}': skewness={target_skew:.3f} → log_transform_target={log_transform}")
    except Exception:
        log_transform = False

    entry: dict = {"features": features, "target": target, "log_transform_target": log_transform, "clip_outliers": False}
    if labels:
        entry["labels"] = labels
    if hints:
        entry["hints"] = hints
    return stem, entry


def collect_labels_hints(features: list[str], stem: str = "") -> tuple[dict, dict]:
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


# ── model_pipeline metric helpers ──────────────────────────────────────────────

def _inverse_transform(arr: np.ndarray, log_transform: bool) -> np.ndarray:
    return np.expm1(arr) if log_transform else arr


def _safe_oof(pred_log: np.ndarray, y_log: np.ndarray, log_transform: bool) -> np.ndarray:
    """Inverse-transform OOF predictions; clips to training range to prevent expm1 blowup."""
    if not log_transform:
        return pred_log
    lo, hi = float(y_log.min()) - 2.0, float(y_log.max()) + 2.0
    return np.expm1(np.clip(pred_log, lo, hi))


def _mmre_pred25(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Returns (MMRE, PRED25). Samples where y_true == 0 are skipped."""
    mask = y_true != 0
    if not mask.any():
        return float("nan"), float("nan")
    rel_err = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    return float(np.mean(rel_err)), float(np.mean(rel_err <= 0.25))


# ── model_pipeline training functions ─────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _iqr_clip(arr: np.ndarray) -> np.ndarray:
    """Clip values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] per column (or 1-D array)."""
    if arr.ndim == 1:
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        return np.clip(arr, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    clipped = arr.copy()
    for j in range(arr.shape[1]):
        q1, q3 = np.percentile(arr[:, j], 25), np.percentile(arr[:, j], 75)
        iqr = q3 - q1
        clipped[:, j] = np.clip(arr[:, j], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return clipped


def preprocess(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    column_map: dict | None = None,
    log_transform: bool = False,
    clip_outliers: bool = False,
):
    if column_map:
        df = df.copy()
        for col, mapping in column_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
    df = df[features + [target]].dropna()
    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} usable rows after dropping NaN — "
            "cross-validation membutuhkan minimal 10 baris."
        )
    X = df[features].values.astype(float)
    y = df[target].values.astype(float)
    if clip_outliers:
        X = _iqr_clip(X)
        y = _iqr_clip(y)
    if log_transform:
        y = np.log1p(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, log_transform


def train_rf(X: np.ndarray, y: np.ndarray, kfold: KFold):
    rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        RF_GRID_PARAMS,
        n_iter=30,
        cv=kfold,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    )
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_


def train_xgb(X: np.ndarray, y: np.ndarray, kfold: KFold):
    gs = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        XGB_GRID_PARAMS,
        cv=kfold,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 20,
) -> LSTMModel:
    val_size = max(1, int(len(X) * 0.2))
    X_tr, X_val = X[:-val_size], X[-val_size:]
    y_tr, y_val = y[:-val_size], y[-val_size:]

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    X_tr_t = to_tensor(X_tr).unsqueeze(1)
    y_tr_t = to_tensor(y_tr)
    X_val_t = to_tensor(X_val).unsqueeze(1)
    y_val_t = to_tensor(y_val)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=16, shuffle=True
    )

    model = LSTMModel(input_size, hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _lstm_cv_scores_with_params(
    X: np.ndarray, y: np.ndarray, kfold: KFold, hidden_size: int, num_layers: int, lr: float,
    log_transform: bool = False,
) -> tuple[float, float, float, float, float]:
    r2_scores = []
    all_y_true: list[float] = []
    all_y_pred: list[float] = []
    for train_idx, val_idx in kfold.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        torch.manual_seed(42)
        m = train_lstm(X_tr, y_tr, input_size=X_tr.shape[1], hidden_size=hidden_size, num_layers=num_layers, lr=lr)
        with torch.no_grad():
            preds = m(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)).numpy()
        ss_res = np.sum((y_val - preds) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(preds.flatten().tolist())
    arr_true = _safe_oof(np.array(all_y_true), y, log_transform)
    arr_pred = _safe_oof(np.array(all_y_pred), y, log_transform)
    mmre, pred25 = _mmre_pred25(arr_true, arr_pred)
    mae = float(np.mean(np.abs(arr_true - arr_pred)))
    rmse = float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))
    r2 = float(r2_score(arr_true, arr_pred)) if len(arr_true) > 1 else 0.0
    return mae, r2, mmre, pred25, rmse


def train_lstm_with_tuning(
    X: np.ndarray, y: np.ndarray, input_size: int, kfold: KFold, log_transform: bool = False,
) -> tuple:
    best_mae = float("inf")
    best_r2 = 0.0
    best_mmre = float("nan")
    best_pred25 = float("nan")
    best_rmse = float("nan")
    best_params: dict = {"hidden_size": 64, "num_layers": 1, "lr": 1e-3}
    for hidden_size in LSTM_GRID_PARAMS["hidden_size"]:
        for num_layers in LSTM_GRID_PARAMS["num_layers"]:
            for lr in LSTM_GRID_PARAMS["lr"]:
                cv_mae, cv_r2, cv_mmre, cv_pred25, cv_rmse = _lstm_cv_scores_with_params(
                    X, y, kfold, hidden_size, num_layers, lr, log_transform=log_transform,
                )
                if cv_mae < best_mae:
                    best_mae = cv_mae
                    best_r2 = cv_r2
                    best_mmre = cv_mmre
                    best_pred25 = cv_pred25
                    best_rmse = cv_rmse
                    best_params = {"hidden_size": hidden_size, "num_layers": num_layers, "lr": lr}
    torch.manual_seed(42)
    model = train_lstm(X, y, input_size, **best_params)
    return model, best_params, best_mae, best_r2, best_mmre, best_pred25, best_rmse


def evaluate_models(X: np.ndarray, y: np.ndarray, input_size: int, log_transform: bool = False) -> dict:
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
    _, rf_best_params = train_rf(X, y, kfold)
    rf_estimator = RandomForestRegressor(random_state=42, **rf_best_params)
    rf_cv = cross_validate(rf_estimator, X, y, cv=kfold, scoring=scoring)
    rf_oof = _safe_oof(cross_val_predict(rf_estimator, X, y, cv=kfold), y, log_transform)
    y_orig = _inverse_transform(y, log_transform)
    rf_mmre, rf_pred25 = _mmre_pred25(y_orig, rf_oof)
    rf_rmse = float(np.sqrt(np.mean((y_orig - rf_oof) ** 2)))

    _, xgb_best_params = train_xgb(X, y, kfold)
    xgb_estimator = XGBRegressor(random_state=42, verbosity=0, **xgb_best_params)
    xgb_cv = cross_validate(xgb_estimator, X, y, cv=kfold, scoring=scoring)
    xgb_oof = _safe_oof(cross_val_predict(xgb_estimator, X, y, cv=kfold), y, log_transform)
    xgb_mmre, xgb_pred25 = _mmre_pred25(y_orig, xgb_oof)
    xgb_rmse = float(np.sqrt(np.mean((y_orig - xgb_oof) ** 2)))

    _, lstm_best_params, lstm_cv_mae, lstm_cv_r2, lstm_mmre, lstm_pred25, lstm_rmse = (
        train_lstm_with_tuning(X, y, input_size, kfold, log_transform=log_transform)
    )

    svr_oof = _safe_oof(cross_val_predict(gs.best_estimator_, X, y, cv=kfold), y, log_transform)
    svr_mmre, svr_pred25 = _mmre_pred25(y_orig, svr_oof)
    svr_rmse = float(np.sqrt(np.mean((y_orig - svr_oof) ** 2)))

    lr_oof = _safe_oof(cross_val_predict(LinearRegression(), X, y, cv=kfold), y, log_transform)
    lr_mmre, lr_pred25 = _mmre_pred25(y_orig, lr_oof)
    lr_rmse = float(np.sqrt(np.mean((y_orig - lr_oof) ** 2)))

    if log_transform:
        svr_mae = float(mean_absolute_error(y_orig, svr_oof))
        lr_mae = float(mean_absolute_error(y_orig, lr_oof))
        rf_mae = float(mean_absolute_error(y_orig, rf_oof))
        xgb_mae = float(mean_absolute_error(y_orig, xgb_oof))
    else:
        svr_mae = float(-svr_cv["test_mae"].mean())
        lr_mae = float(-lr_cv["test_mae"].mean())
        rf_mae = float(-rf_cv["test_mae"].mean())
        xgb_mae = float(-xgb_cv["test_mae"].mean())
    lstm_mae = float(lstm_cv_mae)

    svr_r2 = float(r2_score(y_orig, svr_oof))
    lr_r2 = float(r2_score(y_orig, lr_oof))
    rf_r2 = float(r2_score(y_orig, rf_oof))
    xgb_r2 = float(r2_score(y_orig, xgb_oof))

    w_lstm = 1.0 / max(lstm_mae, 1e-6)
    w_xgb = 1.0 / max(xgb_mae, 1e-6)
    w_lr = 1.0 / max(lr_mae, 1e-6)
    total_w = w_lstm + w_xgb + w_lr
    hybrid_weights = {
        "lstm": w_lstm / total_w,
        "xgb": w_xgb / total_w,
        "lr": w_lr / total_w,
    }

    return {
        "best_svr_params": gs.best_params_,
        "cv_mae_mean": svr_mae,
        "cv_mae_std": float(svr_cv["test_mae"].std()),
        "cv_rmse_mean": svr_rmse,
        "cv_r2_mean": svr_r2,
        "cv_r2_std": float(svr_cv["test_r2"].std()),
        "svr_mmre": svr_mmre,
        "svr_pred25": svr_pred25,
        "lr_cv_mae_mean": lr_mae,
        "lr_cv_rmse_mean": lr_rmse,
        "lr_cv_r2_mean": lr_r2,
        "lr_mmre": lr_mmre,
        "lr_pred25": lr_pred25,
        "rf_cv_mae_mean": rf_mae,
        "rf_cv_rmse_mean": rf_rmse,
        "rf_cv_r2_mean": rf_r2,
        "rf_best_params": rf_best_params,
        "rf_mmre": rf_mmre,
        "rf_pred25": rf_pred25,
        "xgb_cv_mae_mean": xgb_mae,
        "xgb_cv_rmse_mean": xgb_rmse,
        "xgb_cv_r2_mean": xgb_r2,
        "xgb_best_params": xgb_best_params,
        "xgb_mmre": xgb_mmre,
        "xgb_pred25": xgb_pred25,
        "lstm_cv_mae_mean": lstm_mae,
        "lstm_cv_rmse_mean": lstm_rmse,
        "lstm_cv_r2_mean": lstm_cv_r2,
        "lstm_best_params": lstm_best_params,
        "lstm_mmre": lstm_mmre,
        "lstm_pred25": lstm_pred25,
        "hybrid_weights": hybrid_weights,
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
        print(f"          Run 'python pipeline/build.py --configure' to add a configuration for this file.")
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

    secondary = config.get("secondary_target")
    features = [f for f in config["features"] if f != secondary]

    log_transform = bool(config.get("log_transform_target", False))
    clip_outliers = bool(config.get("clip_outliers", False))
    try:
        X, y, scaler, log_transform = preprocess(
            df, features, config["target"],
            column_map=config.get("column_map"),
            log_transform=log_transform,
            clip_outliers=clip_outliers,
        )
    except KeyError as exc:
        print(f"[WARNING] Column {exc} not found in '{os.path.basename(csv_path)}' — skipping.")
        print(f"          Check 'features' and 'target' in '{CONFIG_FILE}' for '{stem}'.")
        return False
    except ValueError as exc:
        print(f"[WARNING] {exc} — skipping '{stem}'.")
        return False

    input_size = X.shape[1]
    print(f"  [{stem}] Menjalankan 5-fold CV + GridSearchCV ({len(y)} baris, {input_size} fitur) ...")
    metrics = evaluate_models(X, y, input_size, log_transform=log_transform)
    metrics["trained_on_rows"] = int(len(y))
    metrics["effort_unit"] = config.get("effort_unit", "person-months")
    metrics["log_transform_target"] = log_transform

    print(f"  [{stem}] SVR  CV-MAE={metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f}  CV-R²={metrics['cv_r2_mean']:.4f}")
    print(f"  [{stem}] LR   CV-MAE={metrics['lr_cv_mae_mean']:.2f}  CV-R²={metrics['lr_cv_r2_mean']:.4f}")
    print(f"  [{stem}] RF   CV-MAE={metrics['rf_cv_mae_mean']:.2f}  CV-R²={metrics['rf_cv_r2_mean']:.4f}  params={metrics['rf_best_params']}")
    print(f"  [{stem}] XGB  CV-MAE={metrics['xgb_cv_mae_mean']:.2f}  CV-R²={metrics['xgb_cv_r2_mean']:.4f}  params={metrics['xgb_best_params']}")
    print(f"  [{stem}] LSTM CV-MAE={metrics['lstm_cv_mae_mean']:.2f}  CV-R²={metrics['lstm_cv_r2_mean']:.4f}  params={metrics['lstm_best_params']}")
    hw = metrics["hybrid_weights"]
    print(f"  [{stem}] Hybrid weights — lstm={hw['lstm']:.3f}  xgb={hw['xgb']:.3f}  lr={hw['lr']:.3f}")
    print(f"  [{stem}] Best SVR params: {metrics['best_svr_params']}")

    final_svr = SVR(**metrics["best_svr_params"])
    final_svr.fit(X, y)
    final_lr = LinearRegression()
    final_lr.fit(X, y)
    final_rf = RandomForestRegressor(random_state=42, **metrics["rf_best_params"])
    final_rf.fit(X, y)
    final_xgb, _ = train_xgb(X, y, KFold(n_splits=5, shuffle=True, random_state=42))
    print(f"  [{stem}] Training final LSTM on full data ...")
    torch.manual_seed(42)
    final_lstm = train_lstm(X, y, input_size, **metrics["lstm_best_params"])

    lstm_hidden = metrics["lstm_best_params"]["hidden_size"]
    lstm_num_layers = metrics["lstm_best_params"].get("num_layers", 1)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_svr, os.path.join(MODELS_DIR, f"svr_{stem}.pkl"))
    joblib.dump(final_lr, os.path.join(MODELS_DIR, f"linreg_{stem}.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{stem}.pkl"))
    joblib.dump(final_rf, os.path.join(MODELS_DIR, f"rf_{stem}.pkl"))
    joblib.dump(final_xgb, os.path.join(MODELS_DIR, f"xgb_{stem}.pkl"))
    torch.save(final_lstm.state_dict(), os.path.join(MODELS_DIR, f"lstm_{stem}.pt"))
    with open(os.path.join(MODELS_DIR, f"lstm_arch_{stem}.json"), "w") as f:
        json.dump({"input_size": input_size, "hidden_size": lstm_hidden, "num_layers": lstm_num_layers}, f)
    save_metrics(stem, metrics)

    print(f"  [{stem}] Exported: svr | linreg | scaler | rf | xgb | lstm | lstm_arch | metrics")
    return True


def cleanup_stale_models(trained_stems: set) -> int:
    existing_svr = glob.glob(os.path.join(MODELS_DIR, "svr_*.pkl"))
    removed = 0
    for svr_path in existing_svr:
        stem = os.path.basename(svr_path)[4:-4]
        if stem not in trained_stems:
            for prefix in ("svr_", "linreg_", "scaler_", "rf_", "xgb_"):
                p = os.path.join(MODELS_DIR, f"{prefix}{stem}.pkl")
                if os.path.exists(p):
                    os.remove(p)
            for name in (f"lstm_{stem}.pt", f"lstm_arch_{stem}.json", f"{stem}_metrics.json"):
                p = os.path.join(MODELS_DIR, name)
                if os.path.exists(p):
                    os.remove(p)
            print(f"  [CLEANUP] Removed stale models for '{stem}'.")
            removed += 1
    return removed


def run_batch() -> tuple[int, set]:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        print(f"[WARNING] No CSV files found in '{DATA_DIR}/'. Run 'python pipeline/downloader.py' first.")
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


# ── Public entry points ────────────────────────────────────────────────────────

def configure() -> None:
    """Interactively configure new datasets found in data/."""
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    if not csv_files:
        print(f"[WARNING] Tidak ada file CSV di '{DATA_DIR}/'. Letakkan CSV di sana lalu jalankan ulang.")
        sys.exit(0)

    print(f"[INFO] Ditemukan {len(csv_files)} file CSV di '{DATA_DIR}/'.")

    existing = load_config()
    original_keys = set(existing.keys())
    if existing:
        print(f"[INFO] Konfigurasi yang sudah ada: {', '.join(existing.keys())}")

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
            print(f"     Jalankan 'python pipeline/build.py --train' untuk melatih model.")
        if backfilled > 0:
            print(f"[OK] {backfilled} dataset diperbarui dengan label/hint/effort_unit default.")
    else:
        print("\n[INFO] Tidak ada perubahan pada konfigurasi.")


def train() -> None:
    """Train models for all configured datasets in data/."""
    total = len(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    succeeded, trained_stems = run_batch()
    if total > 0:
        removed = cleanup_stale_models(trained_stems)
        print(f"\n[OK] Done. {succeeded}/{total} datasets trained successfully.")
        if removed:
            print(f"     Removed {removed} stale model set(s) from '{MODELS_DIR}/'.")
        print(f"     Models saved to '{MODELS_DIR}/'.")


def build_all() -> None:
    """Run configure() then train() in sequence."""
    configure()
    train()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build pipeline: configure datasets and/or train models."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--configure", action="store_true",
        help="Hanya konfigurasi dataset baru (tanpa training)"
    )
    group.add_argument(
        "--train", action="store_true",
        help="Hanya latih model dari konfigurasi yang ada"
    )
    args = parser.parse_args()

    if args.configure:
        configure()
    elif args.train:
        train()
    else:
        build_all()
