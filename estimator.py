import glob
import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch
import torch.nn as nn

HOURS_PER_MONTH = 160

_MODEL_MAE_KEYS: dict[str, str] = {
    "SVR": "cv_mae_mean",
    "Linear Regression": "lr_cv_mae_mean",
    "Random Forest": "rf_cv_mae_mean",
    "XGBoost": "xgb_cv_mae_mean",
    "LSTM": "lstm_cv_mae_mean",
}
_MODEL_MMRE_KEYS: dict[str, str] = {
    "SVR": "svr_mmre",
    "Linear Regression": "lr_mmre",
    "Random Forest": "rf_mmre",
    "XGBoost": "xgb_mmre",
    "LSTM": "lstm_mmre",
}
_MODEL_PRED25_KEYS: dict[str, str] = {
    "SVR": "svr_pred25",
    "Linear Regression": "lr_pred25",
    "Random Forest": "rf_pred25",
    "XGBoost": "xgb_pred25",
    "LSTM": "lstm_pred25",
}
_MODEL_HYPERPARAMS_KEYS: dict[str, str | None] = {
    "SVR": "best_svr_params",
    "Linear Regression": None,
    "Random Forest": "rf_best_params",
    "XGBoost": "xgb_best_params",
    "LSTM": "lstm_best_params",
}


# ── Model definition ──────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
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
    lstm: LSTMModel | None = None
    hybrid_weights: dict | None = field(default=None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_person_months(value: float, unit: str) -> float:
    if unit == "person-hours":
        return value / HOURS_PER_MONTH
    return value


# ── Public API ────────────────────────────────────────────────────────────────

def select_best_estimator(metrics_dir: str) -> EstimatorResult | None:
    """Read all *_metrics.json files and return the best dataset/model pair by CV MAE
    (normalized to person-months). MMRE is used as a tiebreaker within 1% MAE."""
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

    candidates.sort(key=lambda r: r.cv_mae_pm)
    best = candidates[0]
    for c in candidates[1:]:
        if best.cv_mae_pm <= 0:
            break
        gap = (c.cv_mae_pm - best.cv_mae_pm) / best.cv_mae_pm
        if gap < 0.01 and c.mmre is not None and (best.mmre is None or c.mmre < best.mmre):
            best = c
        elif gap >= 0.01:
            break
    return best


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

    arch_path = os.path.join(models_dir, f"lstm_arch_{stem}.json")
    pt_path = os.path.join(models_dir, f"lstm_{stem}.pt")
    if os.path.exists(arch_path) and os.path.exists(pt_path):
        with open(arch_path, "r") as f:
            arch = json.load(f)
        lstm = LSTMModel(arch["input_size"], arch.get("hidden_size", 64))
        lstm.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        lstm.eval()
        m.lstm = lstm

    metrics_path = os.path.join(models_dir, f"{stem}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        m.hybrid_weights = metrics.get("hybrid_weights")

    return m


def run_estimate(
    model_name: str,
    X_scaled: np.ndarray,
    models: LoadedModels,
    unit: str,
) -> float | None:
    """Run a single model prediction; return result in person-months."""
    try:
        if model_name == "SVR":
            if models.svr is None:
                return None
            return _to_person_months(float(models.svr.predict(X_scaled)[0]), unit)
        if model_name == "Linear Regression":
            if models.lr is None:
                return None
            return _to_person_months(float(models.lr.predict(X_scaled)[0]), unit)
        if model_name == "Random Forest":
            if models.rf is None:
                return None
            return _to_person_months(float(models.rf.predict(X_scaled)[0]), unit)
        if model_name == "XGBoost":
            if models.xgb is None:
                return None
            return _to_person_months(float(models.xgb.predict(X_scaled)[0]), unit)
        if model_name == "LSTM":
            if models.lstm is None:
                return None
            xt = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                raw = float(models.lstm(xt).item())
            return _to_person_months(raw, unit)
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

    raw: dict[str, float] = {}
    try:
        if models.lr is not None and weights.get("lr", 0) > 0:
            raw["lr"] = float(models.lr.predict(X_scaled)[0])
        if models.xgb is not None and weights.get("xgb", 0) > 0:
            raw["xgb"] = float(models.xgb.predict(X_scaled)[0])
        if models.lstm is not None and weights.get("lstm", 0) > 0:
            xt = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                raw["lstm"] = float(models.lstm(xt).item())
    except Exception:
        pass

    if not raw:
        return None

    total_w = sum(weights.get(k, 0) for k in raw)
    if total_w <= 0:
        return None

    combined = sum(weights[k] * v for k, v in raw.items()) / total_w
    return _to_person_months(combined, unit)
