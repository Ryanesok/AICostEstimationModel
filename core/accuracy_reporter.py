"""Surface model quality metrics to the PM in a readable summary.

SRS requires MAPE < 10% (mmre < 0.10 = Excellent).
Warning threshold: mmre > 0.15 (MAPE > 15%).
"""
from __future__ import annotations

from core.estimator_bridge import BridgeResult

_WARN_THRESHOLD = 0.15   # SRS: MAPE < 10% ideal; warn at 15%
_EXCELLENT_THRESHOLD = 0.10
_GOOD_THRESHOLD = 0.25


def summarize(result: BridgeResult) -> dict:
    """Return PM-readable model quality summary for a BridgeResult.

    Keys: model_name, dataset, trained_on, mape_pct, mae_pm,
          pred25_pct, quality_label, warn, warn_message
    """
    info = result.model_info
    mmre = info.mmre
    pred25 = info.pred25

    mape_pct = round(mmre * 100.0, 1) if mmre is not None else None
    pred25_pct = round((pred25 or 0.0) * 100.0, 1)

    # Quality label from mmre
    if mmre is None:
        quality_label = "Unknown"
        warn = True
        warn_message = "Model quality metrics unavailable — treat estimate as indicative only."
    elif mmre <= _EXCELLENT_THRESHOLD:
        quality_label = "Excellent"
        warn = False
        warn_message = ""
    elif mmre <= _GOOD_THRESHOLD:
        quality_label = "Good"
        warn = mmre > _WARN_THRESHOLD
        warn_message = (
            f"Model MAPE is {mape_pct}% — above the 15% caution threshold. "
            "Consider retraining with more data."
        ) if warn else ""
    else:
        quality_label = "Fair"
        warn = True
        warn_message = (
            f"Model MAPE is {mape_pct}% — significantly above the 15% threshold. "
            "Estimates may be unreliable; retraining recommended."
        )

    return {
        "model_name": info.model_name,
        "dataset": info.stem,
        "trained_on": info.trained_on_rows,
        "mape_pct": mape_pct,
        "mae_pm": round(info.cv_mae_pm, 2),
        "pred25_pct": pred25_pct,
        "quality_label": quality_label,
        "warn": warn,
        "warn_message": warn_message,
    }
