import glob
import json
import math
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch
import torch.nn as nn

HOURS_PER_MONTH = 160
MIN_TRAINING_ROWS = 50
CONFIDENCE_THRESHOLD = 0.75  # minimum PRED(25) for a model to be considered high-confidence

_MODEL_MAE_KEYS: dict[str, str] = {
    "SVR": "cv_mae_mean",
    "Linear Regression": "lr_cv_mae_mean",
    "Random Forest": "rf_cv_mae_mean",
    "XGBoost": "xgb_cv_mae_mean",
    "KNN": "knn_cv_mae_mean",
    "LSTM": "lstm_cv_mae_mean",
    "Hybrid": "hybrid_cv_mae_mean",
}
_MODEL_MMRE_KEYS: dict[str, str] = {
    "SVR": "svr_mmre",
    "Linear Regression": "lr_mmre",
    "Random Forest": "rf_mmre",
    "XGBoost": "xgb_mmre",
    "KNN": "knn_mmre",
    "LSTM": "lstm_mmre",
    "Hybrid": "hybrid_mmre",
}
_MODEL_PRED25_KEYS: dict[str, str] = {
    "SVR": "svr_pred25",
    "Linear Regression": "lr_pred25",
    "Random Forest": "rf_pred25",
    "XGBoost": "xgb_pred25",
    "KNN": "knn_pred25",
    "LSTM": "lstm_pred25",
    "Hybrid": "hybrid_pred25",
}
_MODEL_HYPERPARAMS_KEYS: dict[str, str | None] = {
    "SVR": "best_svr_params",
    "Linear Regression": None,
    "Random Forest": "rf_best_params",
    "XGBoost": "xgb_best_params",
    "KNN": "knn_best_params",
    "LSTM": "lstm_best_params",
    "Hybrid": None,
}


# ── Model definition ──────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EstimatorResult:
    stem: str
    model_name: str
    cv_mae: float           # raw (dataset's native unit)
    cv_mae_pm: float        # normalized to person-months
    mmre: float | None
    pred25: float | None
    hyperparams: dict
    effort_unit: str = "person-months"
    trained_on_rows: int = 0


@dataclass
class LoadedModels:
    svr: object = None
    lr: object = None
    scaler: object = None
    rf: object = None
    xgb: object = None
    knn: object = None
    lstm: LSTMModel | None = None
    hybrid_weights: dict | None = field(default=None)
    log_transform_target: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_person_months(value: float, unit: str) -> float:
    if unit == "person-hours":
        return value / HOURS_PER_MONTH
    return value


# ── Public API ────────────────────────────────────────────────────────────────

def select_best_estimator(
    metrics_dir: str,
    supported_stems: frozenset[str] | None = None,
    min_pred25: float = 0.0,
) -> EstimatorResult | None:
    """Read all *_metrics.json files and return the best dataset/model pair.

    Selection priority (each step falls back to the previous pool if empty):
    1. Remove datasets below MIN_TRAINING_ROWS to avoid overfitting on tiny data.
    2. Restrict to supported_stems when given (app-level filter for registered datasets).
    3. Apply confidence gate: keep only candidates with PRED(25) >= min_pred25.
    4. Among remaining candidates, sort by PRED(25) descending; use CV MAE as tiebreaker
       when PRED(25) values are equal.
    """
    files = glob.glob(os.path.join(metrics_dir, "*_metrics.json"))
    if not files:
        return None

    candidates: list[EstimatorResult] = []
    for path in files:
        stem = os.path.basename(path)[: -len("_metrics.json")]
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        effort_unit = m.get("effort_unit", "person-months")
        rows = m.get("trained_on_rows", 0)

        for model_name, mae_key in _MODEL_MAE_KEYS.items():
            mae = m.get(mae_key)
            if mae is None:
                continue
            mae_pm = _to_person_months(mae, effort_unit)
            mmre = m.get(_MODEL_MMRE_KEYS[model_name])
            pred25 = m.get(_MODEL_PRED25_KEYS[model_name])
            hp_key = _MODEL_HYPERPARAMS_KEYS[model_name]
            hyperparams = m.get(hp_key, {}) if hp_key else {}
            candidates.append(EstimatorResult(
                stem=stem,
                model_name=model_name,
                cv_mae=mae,
                cv_mae_pm=mae_pm,
                mmre=mmre,
                pred25=pred25,
                hyperparams=hyperparams or {},
                effort_unit=effort_unit,
                trained_on_rows=rows,
            ))

    if not candidates:
        return None

    # Prefer datasets with enough training rows; fall back to all if none qualify.
    qualified = [c for c in candidates if c.trained_on_rows >= MIN_TRAINING_ROWS]
    pool = qualified if qualified else candidates

    # Restrict to supported stems when caller provides a filter; fall back to full
    # pool if no candidates match so the app stays functional.
    if supported_stems:
        supported_pool = [c for c in pool if c.stem in supported_stems]
        if supported_pool:
            pool = supported_pool

    # Confidence gate: prefer models with PRED(25) >= min_pred25.
    # Falls back to full pool if no candidate meets the threshold.
    if min_pred25 > 0:
        confident_pool = [c for c in pool if c.pred25 is not None and c.pred25 >= min_pred25]
        if confident_pool:
            pool = confident_pool

    # Primary: highest PRED(25) first; secondary: lowest MAE (person-months).
    pool.sort(key=lambda r: (-(r.pred25 or 0), r.cv_mae_pm))
    return pool[0]


def load_estimator_models(stem: str, models_dir: str) -> LoadedModels:
    """Load all model artefacts for the given dataset stem."""
    m = LoadedModels()
    try:
        m.svr = joblib.load(os.path.join(models_dir, f"svr_{stem}.pkl"))
        m.lr = joblib.load(os.path.join(models_dir, f"linreg_{stem}.pkl"))
        m.scaler = joblib.load(os.path.join(models_dir, f"scaler_{stem}.pkl"))
    except FileNotFoundError:
        return m

    rf_path = os.path.join(models_dir, f"rf_{stem}.pkl")
    if os.path.exists(rf_path):
        m.rf = joblib.load(rf_path)

    xgb_path = os.path.join(models_dir, f"xgb_{stem}.pkl")
    if os.path.exists(xgb_path):
        m.xgb = joblib.load(xgb_path)

    knn_path = os.path.join(models_dir, f"knn_{stem}.pkl")
    if os.path.exists(knn_path):
        m.knn = joblib.load(knn_path)

    arch_path = os.path.join(models_dir, f"lstm_arch_{stem}.json")
    pt_path = os.path.join(models_dir, f"lstm_{stem}.pt")
    if os.path.exists(arch_path) and os.path.exists(pt_path):
        with open(arch_path, "r") as f:
            arch = json.load(f)
        lstm = LSTMModel(arch["input_size"], arch.get("hidden_size", 64), arch.get("num_layers", 1))
        lstm.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        lstm.eval()
        m.lstm = lstm

    metrics_path = os.path.join(models_dir, f"{stem}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        m.hybrid_weights = metrics.get("hybrid_weights")
        m.log_transform_target = bool(metrics.get("log_transform_target", False))

    return m


def run_estimate(
    model_name: str,
    X_scaled: np.ndarray,
    models: LoadedModels,
    unit: str,
) -> float | None:
    """Run a single model prediction; return result in person-months."""
    def _postprocess(raw: float) -> float:
        v = math.expm1(raw) if models.log_transform_target else raw
        return _to_person_months(v, unit)

    try:
        if model_name == "SVR":
            if models.svr is None:
                return None
            return _postprocess(float(models.svr.predict(X_scaled)[0]))
        if model_name == "Linear Regression":
            if models.lr is None:
                return None
            return _postprocess(float(models.lr.predict(X_scaled)[0]))
        if model_name == "Random Forest":
            if models.rf is None:
                return None
            return _postprocess(float(models.rf.predict(X_scaled)[0]))
        if model_name == "XGBoost":
            if models.xgb is None:
                return None
            return _postprocess(float(models.xgb.predict(X_scaled)[0]))
        if model_name == "KNN":
            if models.knn is None:
                return None
            return _postprocess(float(models.knn.predict(X_scaled)[0]))
        if model_name == "LSTM":
            if models.lstm is None:
                return None
            xt = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                raw = float(models.lstm(xt).item())
            return _postprocess(raw)
    except Exception:
        return None
    return None


def run_hybrid_estimate(
    X_scaled: np.ndarray,
    models: LoadedModels,
    unit: str,
) -> float | None:
    """Weighted ensemble of LR, XGB, LSTM using stored hybrid_weights.

    Weights were optimised during training to minimise CV error.
    Any unavailable model is excluded and remaining weights are renormalised.
    Returns person-months, or None if no models are available.
    """
    weights = models.hybrid_weights
    if not weights:
        return None

    # Inverse-transform each model output before averaging (weighted mean in original scale)
    raw: dict[str, float] = {}
    try:
        if models.svr is not None and weights.get("svr", 0) > 0:
            v = float(models.svr.predict(X_scaled)[0])
            raw["svr"] = math.expm1(v) if models.log_transform_target else v
        if models.lr is not None and weights.get("lr", 0) > 0:
            v = float(models.lr.predict(X_scaled)[0])
            raw["lr"] = math.expm1(v) if models.log_transform_target else v
        if models.rf is not None and weights.get("rf", 0) > 0:
            v = float(models.rf.predict(X_scaled)[0])
            raw["rf"] = math.expm1(v) if models.log_transform_target else v
        if models.xgb is not None and weights.get("xgb", 0) > 0:
            v = float(models.xgb.predict(X_scaled)[0])
            raw["xgb"] = math.expm1(v) if models.log_transform_target else v
        if models.knn is not None and weights.get("knn", 0) > 0:
            v = float(models.knn.predict(X_scaled)[0])
            raw["knn"] = math.expm1(v) if models.log_transform_target else v
        if models.lstm is not None and weights.get("lstm", 0) > 0:
            xt = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                v = float(models.lstm(xt).item())
            raw["lstm"] = math.expm1(v) if models.log_transform_target else v
    except Exception:
        pass

    if not raw:
        return None

    total_w = sum(weights.get(k, 0) for k in raw)
    if total_w <= 0:
        return None

    combined = sum(weights[k] * v for k, v in raw.items()) / total_w
    return _to_person_months(combined, unit)
