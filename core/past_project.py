"""Persist and retrieve past estimation records; cosine-similarity lookup."""
from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os

from core.schema import PMInput
from core.estimator_bridge import BridgeResult

_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "last_inputs", "history.json")

# Numeric fields used for similarity (categorical fields are excluded)
_NUMERIC_FIELDS = [
    "num_features", "num_apis", "third_party_integrations",
    "num_developers", "deadline_months", "external_dependencies",
]

# Ordinal encodings for categorical fields included in similarity
_ORDINAL = {
    "feature_complexity":      {"low": 1, "medium": 2, "high": 3},
    "security_level":          {"basic": 1, "standard": 2, "high": 3},
    "seniority":               {"junior": 1, "mid": 2, "senior": 3, "mixed": 2},
    "stack_experience":        {"low": 1, "medium": 2, "high": 3},
    "budget_constraint":       {"none": 1, "soft": 2, "hard": 3},
    "requirement_change_risk": {"low": 1, "medium": 2, "high": 3},
    "uncertainty_level":       {"low": 1, "medium": 2, "high": 3},
    "technical_debt":          {"none": 0, "low": 1, "high": 3},
}


def _to_vector(pm_input: PMInput) -> list[float]:
    vec = [float(getattr(pm_input, f)) for f in _NUMERIC_FIELDS]
    for field, mapping in _ORDINAL.items():
        vec.append(float(mapping.get(getattr(pm_input, field), 1)))
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _load_raw() -> list[dict]:
    if not os.path.exists(_HISTORY_FILE):
        return []
    with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_raw(records: list[dict]) -> None:
    os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# ── Public API ────────────────────────────────────────────────────────────────

def save(pm_input: PMInput, result: BridgeResult, confidence: dict, label: str = "") -> None:
    """Append an estimation record to the history store."""
    records = _load_raw()
    records.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "label": label,
        "pm_input": dataclasses.asdict(pm_input),
        "effort_pm": result.effort_pm,
        "model_name": result.model_info.model_name,
        "dataset": result.model_info.stem,
        "confidence_pct": confidence.get("confidence_pct"),
        "similarity": None,
    })
    _save_raw(records)


def load_all() -> list[dict]:
    """Return all saved estimation records."""
    return _load_raw()


def find_similar(pm_input: PMInput, top_n: int = 3) -> list[dict]:
    """Return top_n past estimates most similar to pm_input (cosine similarity).

    Each returned record includes a 'similarity' key (0–1).
    """
    records = _load_raw()
    if not records:
        return []

    query_vec = _to_vector(pm_input)
    scored = []
    for rec in records:
        try:
            past_pm = PMInput(**rec["pm_input"])
            past_vec = _to_vector(past_pm)
            sim = _cosine_similarity(query_vec, past_vec)
            scored.append({**rec, "similarity": round(sim, 4)})
        except Exception:
            continue

    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:top_n]
