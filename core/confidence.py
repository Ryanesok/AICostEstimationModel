"""Confidence interval and reliability score for a BridgeResult.

Formula rationale
-----------------
We have two model quality signals in EstimatorResult:
  - cv_mae_pm : cross-validated MAE in person-months  (lower = better)
  - mmre      : Mean Magnitude of Relative Error       (lower = better)
  - pred25    : fraction of predictions within 25% of actuals  (higher = better)

Confidence is built from pred25 (primary) and mmre (secondary):
  base_confidence = pred25 * 100          [0-100]
  mmre_penalty    = mmre * 30             [penalty up to ~30 pts for mmre=1.0]
  confidence_pct  = clamp(base - penalty, 0, 100)

If pred25 is unavailable we fall back to a MAE-ratio heuristic:
  fallback = clamp(100 - (cv_mae_pm / effort_pm) * 50, 0, 100)

The interval is a symmetric band around effort_pm using cv_mae_pm as
the half-width, scaled by a z-factor derived from confidence_pct:
  half = cv_mae_pm * (2.0 - confidence_pct / 100)
  lower = effort_pm - half
  upper = effort_pm + half
  range_pct = half / effort_pm * 100  (if effort_pm > 0)
"""
from __future__ import annotations

from core.estimator_bridge import BridgeResult


def compute(result: BridgeResult) -> dict:
    """Compute confidence interval and reliability score for a BridgeResult.

    Returns:
        lower        : lower bound of estimated effort (person-months)
        upper        : upper bound
        confidence_pct : reliability score in [0, 100]
        range_pct    : half-width as a percentage of the central estimate
    """
    effort = result.effort_pm
    info = result.model_info
    mae_pm = info.cv_mae_pm

    # ── Confidence score ──────────────────────────────────────────────────────
    pred25 = info.pred25   # fraction of predictions within 25% of actual
    mmre = info.mmre

    if pred25 is not None:
        base = pred25 * 100.0
        penalty = (mmre * 30.0) if mmre is not None else 0.0
        confidence_pct = base - penalty
    else:
        # Fallback: penalise by MAE relative to the prediction
        if effort > 0:
            confidence_pct = 100.0 - (mae_pm / effort) * 50.0
        else:
            confidence_pct = 50.0

    # Clamp to [0, 100]
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    # ── Interval ──────────────────────────────────────────────────────────────
    # Half-width grows as confidence drops: at 100% → 1x MAE, at 0% → 2x MAE
    half = mae_pm * (2.0 - confidence_pct / 100.0)
    lower = max(0.0, effort - half)
    upper = effort + half

    range_pct = (half / effort * 100.0) if effort > 0 else 0.0

    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "confidence_pct": round(confidence_pct, 1),
        "range_pct": round(range_pct, 1),
    }
