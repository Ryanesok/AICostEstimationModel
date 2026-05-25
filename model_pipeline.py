import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

DATA_DIR = "data"
MODELS_DIR = "models"
CONFIG_FILE = "dataset_config.yaml"

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
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

LSTM_GRID_PARAMS = {
    "hidden_size": [32, 64, 128],
    "lr": [1e-3, 5e-4],
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _mmre_pred25(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Returns (MMRE, PRED25). Samples where y_true == 0 are skipped."""
    mask = y_true != 0
    if not mask.any():
        return float("nan"), float("nan")
    rel_err = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    return float(np.mean(rel_err)), float(np.mean(rel_err <= 0.25))


# ── Model definitions ──────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len=1, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


# ── Config helpers ─────────────────────────────────────────────────────────────

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


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, features: list[str], target: str, column_map: dict | None = None):
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
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


# ── Training functions ─────────────────────────────────────────────────────────

def train_rf(X: np.ndarray, y: np.ndarray, kfold: KFold):
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        RF_GRID_PARAMS,
        cv=kfold,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


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

    model = LSTMModel(input_size, hidden_size)
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


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _lstm_cv_scores_with_params(
    X: np.ndarray, y: np.ndarray, kfold: KFold, hidden_size: int, lr: float
) -> tuple[float, float, float, float]:
    mae_scores, r2_scores = [], []
    all_y_true: list[float] = []
    all_y_pred: list[float] = []
    for train_idx, val_idx in kfold.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        torch.manual_seed(42)
        m = train_lstm(X_tr, y_tr, input_size=X_tr.shape[1], hidden_size=hidden_size, lr=lr)
        with torch.no_grad():
            preds = m(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)).numpy()
        mae_scores.append(mean_absolute_error(y_val, preds))
        ss_res = np.sum((y_val - preds) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(preds.flatten().tolist())
    mmre, pred25 = _mmre_pred25(np.array(all_y_true), np.array(all_y_pred))
    return float(np.mean(mae_scores)), float(np.mean(r2_scores)), mmre, pred25


def train_lstm_with_tuning(
    X: np.ndarray, y: np.ndarray, input_size: int, kfold: KFold
) -> tuple:
    best_mae = float("inf")
    best_r2 = 0.0
    best_mmre = float("nan")
    best_pred25 = float("nan")
    best_params: dict = {"hidden_size": 64, "lr": 1e-3}
    for hidden_size in LSTM_GRID_PARAMS["hidden_size"]:
        for lr in LSTM_GRID_PARAMS["lr"]:
            cv_mae, cv_r2, cv_mmre, cv_pred25 = _lstm_cv_scores_with_params(
                X, y, kfold, hidden_size, lr
            )
            if cv_mae < best_mae:
                best_mae = cv_mae
                best_r2 = cv_r2
                best_mmre = cv_mmre
                best_pred25 = cv_pred25
                best_params = {"hidden_size": hidden_size, "lr": lr}
    torch.manual_seed(42)
    model = train_lstm(X, y, input_size, **best_params)
    return model, best_params, best_mae, best_r2, best_mmre, best_pred25


def evaluate_models(X: np.ndarray, y: np.ndarray, input_size: int) -> dict:
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
    rf_oof = cross_val_predict(rf_estimator, X, y, cv=kfold)
    rf_mmre, rf_pred25 = _mmre_pred25(y, rf_oof)

    _, xgb_best_params = train_xgb(X, y, kfold)
    xgb_estimator = XGBRegressor(random_state=42, verbosity=0, **xgb_best_params)
    xgb_cv = cross_validate(xgb_estimator, X, y, cv=kfold, scoring=scoring)
    xgb_oof = cross_val_predict(xgb_estimator, X, y, cv=kfold)
    xgb_mmre, xgb_pred25 = _mmre_pred25(y, xgb_oof)

    _, lstm_best_params, lstm_cv_mae, lstm_cv_r2, lstm_mmre, lstm_pred25 = (
        train_lstm_with_tuning(X, y, input_size, kfold)
    )

    svr_oof = cross_val_predict(gs.best_estimator_, X, y, cv=kfold)
    svr_mmre, svr_pred25 = _mmre_pred25(y, svr_oof)

    lr_oof = cross_val_predict(LinearRegression(), X, y, cv=kfold)
    lr_mmre, lr_pred25 = _mmre_pred25(y, lr_oof)

    lr_mae = float(-lr_cv["test_mae"].mean())
    xgb_mae = float(-xgb_cv["test_mae"].mean())
    lstm_mae = float(lstm_cv_mae)

    # Hybrid weights: inverse MAE, normalized (LSTM + XGBoost + LR)
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
        "cv_mae_mean": float(-svr_cv["test_mae"].mean()),
        "cv_mae_std": float(svr_cv["test_mae"].std()),
        "cv_r2_mean": float(svr_cv["test_r2"].mean()),
        "cv_r2_std": float(svr_cv["test_r2"].std()),
        "svr_mmre": svr_mmre,
        "svr_pred25": svr_pred25,
        "lr_cv_mae_mean": lr_mae,
        "lr_cv_r2_mean": float(lr_cv["test_r2"].mean()),
        "lr_mmre": lr_mmre,
        "lr_pred25": lr_pred25,
        "rf_cv_mae_mean": float(-rf_cv["test_mae"].mean()),
        "rf_cv_r2_mean": float(rf_cv["test_r2"].mean()),
        "rf_best_params": rf_best_params,
        "rf_mmre": rf_mmre,
        "rf_pred25": rf_pred25,
        "xgb_cv_mae_mean": xgb_mae,
        "xgb_cv_r2_mean": float(xgb_cv["test_r2"].mean()),
        "xgb_best_params": xgb_best_params,
        "xgb_mmre": xgb_mmre,
        "xgb_pred25": xgb_pred25,
        "lstm_cv_mae_mean": lstm_mae,
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


# ── Training + export ──────────────────────────────────────────────────────────

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
        X, y, scaler = preprocess(df, features, config["target"], column_map=config.get("column_map"))
    except KeyError as exc:
        print(f"[WARNING] Column {exc} not found in '{os.path.basename(csv_path)}' — skipping.")
        print(f"          Check 'features' and 'target' in '{CONFIG_FILE}' for '{stem}'.")
        return False
    except ValueError as exc:
        print(f"[WARNING] {exc} — skipping '{stem}'.")
        return False

    input_size = X.shape[1]
    print(f"  [{stem}] Menjalankan 5-fold CV + GridSearchCV ({len(y)} baris, {input_size} fitur) ...")
    metrics = evaluate_models(X, y, input_size)
    metrics["trained_on_rows"] = int(len(y))
    metrics["effort_unit"] = config.get("effort_unit", "person-months")

    print(f"  [{stem}] SVR  CV-MAE={metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f}  CV-R²={metrics['cv_r2_mean']:.4f}")
    print(f"  [{stem}] LR   CV-MAE={metrics['lr_cv_mae_mean']:.2f}  CV-R²={metrics['lr_cv_r2_mean']:.4f}")
    print(f"  [{stem}] RF   CV-MAE={metrics['rf_cv_mae_mean']:.2f}  CV-R²={metrics['rf_cv_r2_mean']:.4f}  params={metrics['rf_best_params']}")
    print(f"  [{stem}] XGB  CV-MAE={metrics['xgb_cv_mae_mean']:.2f}  CV-R²={metrics['xgb_cv_r2_mean']:.4f}  params={metrics['xgb_best_params']}")
    print(f"  [{stem}] LSTM CV-MAE={metrics['lstm_cv_mae_mean']:.2f}  CV-R²={metrics['lstm_cv_r2_mean']:.4f}  params={metrics['lstm_best_params']}")
    hw = metrics["hybrid_weights"]
    print(f"  [{stem}] Hybrid weights — lstm={hw['lstm']:.3f}  xgb={hw['xgb']:.3f}  lr={hw['lr']:.3f}")
    print(f"  [{stem}] Best SVR params: {metrics['best_svr_params']}")

    # Train final models on full data using best params found during evaluation
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
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_svr, os.path.join(MODELS_DIR, f"svr_{stem}.pkl"))
    joblib.dump(final_lr, os.path.join(MODELS_DIR, f"linreg_{stem}.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{stem}.pkl"))
    joblib.dump(final_rf, os.path.join(MODELS_DIR, f"rf_{stem}.pkl"))
    joblib.dump(final_xgb, os.path.join(MODELS_DIR, f"xgb_{stem}.pkl"))
    torch.save(final_lstm.state_dict(), os.path.join(MODELS_DIR, f"lstm_{stem}.pt"))
    with open(os.path.join(MODELS_DIR, f"lstm_arch_{stem}.json"), "w") as f:
        json.dump({"input_size": input_size, "hidden_size": lstm_hidden}, f)
    save_metrics(stem, metrics)

    print(
        f"  [{stem}] Exported: svr | linreg | scaler | rf | xgb | lstm | lstm_arch | metrics"
    )
    return True


# ── Cleanup ────────────────────────────────────────────────────────────────────

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
            print(f"     Removed {removed} stale model set(s) from '{MODELS_DIR}/'.")
        print(f"     Models saved to '{MODELS_DIR}/'.")
