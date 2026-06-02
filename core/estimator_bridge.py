"""Bridge between PM-facing PMInput and the dataset-trained ML pipeline.

Each supported dataset requires a specific feature vector. This module
maintains one mapping function per dataset, selected at runtime based on
whichever model scored best during training.

Effort unit output: always person-months (pipeline normalises internally).
"""
from __future__ import annotations

import copy
import dataclasses
import os

import numpy as np

from core.schema import PMInput
from pipeline.estimator import (
    CONFIDENCE_THRESHOLD,
    EstimatorResult,
    load_estimator_models,
    run_estimate,
    run_hybrid_estimate,
    select_best_estimator,
)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Ordinal mappings shared across dataset functions
_COMPLEXITY_SCORE = {"low": 1.0, "medium": 1.5, "high": 2.5}
_SENIORITY_YEARS = {"junior": 1.0, "mid": 3.0, "senior": 6.0, "mixed": 3.5}
_STACK_SCORE = {"low": 1.0, "medium": 2.0, "high": 3.0}
_SECURITY_FACTOR = {"basic": 1.0, "standard": 1.15, "high": 1.35}
_RISK_SCORE = {"low": 1.0, "medium": 2.0, "high": 3.0}
_DEBT_SCORE = {"none": 0.0, "low": 1.0, "high": 2.5}


@dataclasses.dataclass
class BridgeResult:
    """Prediction result produced by the bridge.

    effort_pm  : actual model prediction for this PMInput (person-months)
    model_info : best-model metadata (cv_mae, mmre, pred25, etc.)
    """
    effort_pm: float
    model_info: EstimatorResult


def _resource_level(num_developers: int) -> float:
    """Maps developer count to a 1-5 resource level ordinal."""
    if num_developers <= 2:
        return 1.0
    if num_developers <= 4:
        return 2.0
    if num_developers <= 8:
        return 3.0
    if num_developers <= 15:
        return 4.0
    return 5.0


def _synthetic_fp(pm: PMInput) -> float:
    """Synthetic Function Point score derived from PM inputs.

    FP = (features x complexity_multiplier x 5) + (APIs x 3) +
         (integrations x 2), then scaled by a security overhead factor.
    """
    complexity = _COMPLEXITY_SCORE[pm.feature_complexity]
    raw_fp = (pm.num_features * complexity * 5.0) + (pm.num_apis * 3.0) + (pm.third_party_integrations * 2.0)
    return raw_fp * _SECURITY_FACTOR[pm.security_level]


# ── Per-dataset feature vector builders ───────────────────────────────────────

def _features_desharnais(pm: PMInput) -> list[float]:
    """desharnais features: [TeamExp, ManagerExp, Transactions, Entities]

    TeamExp      <- seniority proxy in years (junior=1, mid=3, senior=6, mixed=3.5)
    ManagerExp   <- fixed 5; PM experience is not a PM input
    Transactions <- num_features x complexity + num_apis (system transaction count proxy)
    Entities     <- num_features x 0.5 (roughly half the features map to data entities)
    """
    return [
        _SENIORITY_YEARS[pm.seniority],
        5.0,
        pm.num_features * _COMPLEXITY_SCORE[pm.feature_complexity] + pm.num_apis,
        pm.num_features * 0.5,
    ]


def _features_china(pm: PMInput) -> list[float]:
    """china features: [AFP, Input, Output, Enquiry, File, Interface,
                        Added, Changed, Deleted, Resource]

    AFP       <- synthetic FP adjusted for security
    Input     <- 30% of features (input-heavy use cases)
    Output    <- 20% of features
    Enquiry   <- 50% of APIs (read-only endpoints)
    File      <- 15% of features (internal logical files)
    Interface <- third_party_integrations
    Added     <- num_features (assume all-new project; Changed=Deleted=0)
    Resource  <- ordinal 1-5 from num_developers
    """
    fp = _synthetic_fp(pm)
    return [
        fp,
        pm.num_features * 0.30,
        pm.num_features * 0.20,
        pm.num_apis * 0.50,
        pm.num_features * 0.15,
        float(pm.third_party_integrations),
        float(pm.num_features),
        0.0,
        0.0,
        _resource_level(pm.num_developers),
    ]


def _features_kitchenham(pm: PMInput) -> list[float]:
    """kitchenham features: [Adjusted.function.points, First.estimate]

    AFP           <- synthetic FP
    First.estimate<- rough person-hours proxy: AFP x 8 hrs/FP, adjusted by
                     seniority (senior teams deliver faster: scale by exp/3)
    """
    fp = _synthetic_fp(pm)
    hrs_per_fp = 8.0 / (_SENIORITY_YEARS[pm.seniority] / 3.0)
    return [fp, fp * hrs_per_fp]


def _features_subbiah(pm: PMInput) -> list[float]:
    """subbiah features: [TotalUFP, ExternalInterface, ExternalInputs,
                          ExternalOutputs, ExternalQueries, InternalLogicalFiles,
                          Adjustment]

    TotalUFP      <- num_features x complexity x 5 (unadjusted FP)
    ExternalInterface <- third_party_integrations
    ExternalInputs    <- 30% of features
    ExternalOutputs   <- 25% of features
    ExternalQueries   <- 50% of APIs
    InternalLogicalFiles <- 15% of features
    Adjustment    <- security_factor as proxy for Value Adjustment Factor
    """
    ufp = pm.num_features * _COMPLEXITY_SCORE[pm.feature_complexity] * 5.0
    return [
        ufp,
        float(pm.third_party_integrations),
        pm.num_features * 0.30,
        pm.num_features * 0.25,
        pm.num_apis * 0.50,
        pm.num_features * 0.15,
        _SECURITY_FACTOR[pm.security_level],
    ]


def _features_isbsg10(pm: PMInput) -> list[float]:
    """isbsg10 features: [FS, N_PDR, Resource_Level]

    FS           <- synthetic FP (functional size)
    N_PDR        <- normalized delivery rate (effort hours per FP), derived from
                    seniority (higher experience = lower rate) and risk multiplier
    Resource_Level <- ordinal 1-5 from num_developers
    """
    fp = _synthetic_fp(pm)
    risk_mult = _RISK_SCORE[pm.uncertainty_level] * (1.0 + _DEBT_SCORE[pm.technical_debt] * 0.1)
    base_pdr = 8.0 / _SENIORITY_YEARS[pm.seniority]  # hours per FP at this seniority
    n_pdr = base_pdr * risk_mult
    return [fp, n_pdr, _resource_level(pm.num_developers)]


def _features_kemerer(pm: PMInput) -> list[float]:
    """kemerer features: [Hardware, KSLOC, AdjFP, RAWFP]

    Hardware <- platform ordinal: web=1, mobile=2, desktop=3, embedded=4, other=5
    KSLOC    <- KLOC proxy: features x complexity x 1.5  (avg 1.5K lines per feature)
    AdjFP    <- synthetic AFP (adjusted for security)
    RAWFP    <- FP without security adjustment
    """
    _platform_hw = {"web": 1.0, "mobile": 2.0, "desktop": 3.0, "embedded": 4.0, "other": 5.0}
    hardware = _platform_hw.get(pm.platform.lower(), 5.0)
    complexity = _COMPLEXITY_SCORE[pm.feature_complexity]
    ksloc = pm.num_features * complexity * 1.5
    afp = _synthetic_fp(pm)
    raw_fp = (pm.num_features * complexity * 5.0) + (pm.num_apis * 3.0) + (pm.third_party_integrations * 2.0)
    return [hardware, ksloc, afp, raw_fp]


def _features_nasa93(pm: PMInput) -> list[float]:
    """nasa93 features (COCOMO-II cost drivers), order matches training data:
    [mode, rely, data, cplx, time, stor, virt, turn, acap, aexp, pcap, vexp,
     lexp, modp, tool, sced, equivphyskloc]

    COCOMO ordinal multipliers are mapped from PMInput categoricals.
    Constraints without a direct PMInput proxy use their nominal (1.0) value.
    """
    # mode: 1=organic (simple, small team), 2=semi-detached, 3=embedded (complex)
    _mode_map = {"low": 1.0, "medium": 2.0, "high": 3.0}
    mode = _mode_map[pm.feature_complexity]

    # Personnel rating multipliers (lower = more capable)
    _acap_map = {"junior": 1.00, "mid": 0.86, "senior": 0.71, "mixed": 0.86}
    _aexp_map = {"junior": 1.13, "mid": 1.00, "senior": 0.91, "mixed": 1.00}
    _pcap_map = {"junior": 1.17, "mid": 0.86, "senior": 0.70, "mixed": 0.86}
    _vexp_map = {"low": 1.10, "medium": 1.00, "high": 0.90}
    _lexp_map = {"low": 1.07, "medium": 1.00, "high": 0.95}

    # rely: required reliability (security level as proxy)
    _rely_map = {"basic": 0.88, "standard": 1.00, "high": 1.15}
    rely = _rely_map[pm.security_level]

    # cplx: product complexity
    _cplx_map = {"low": 0.85, "medium": 1.15, "high": 1.65}
    cplx = _cplx_map[pm.feature_complexity]

    # sced: required development schedule constraint
    # (shorter deadline relative to nominal → higher multiplier)
    sced = 1.08 if pm.deadline_months < 6 else (1.04 if pm.deadline_months < 12 else 1.00)

    # equivphyskloc: estimated size (features × complexity_factor × 1.5K LoC per feature)
    kloc = pm.num_features * _COMPLEXITY_SCORE[pm.feature_complexity] * 1.5

    return [
        mode,                           # mode
        rely,                           # rely
        1.00,                           # data (nominal — no PMInput proxy)
        cplx,                           # cplx
        1.00,                           # time (nominal)
        1.00,                           # stor (nominal)
        1.00,                           # virt (nominal)
        1.00,                           # turn (nominal)
        _acap_map[pm.seniority],        # acap
        _aexp_map[pm.seniority],        # aexp
        _pcap_map[pm.seniority],        # pcap
        _vexp_map[pm.stack_experience], # vexp
        _lexp_map[pm.stack_experience], # lexp
        1.00,                           # modp (nominal)
        1.00,                           # tool (nominal)
        sced,                           # sced
        kloc,                           # equivphyskloc
    ]


def _features_miyazaki94(pm: PMInput) -> list[float]:
    """miyazaki94 features: [KLOC, SCRN, FORM, FILE, ESCRN, EFORM, EFILE]

    48 Japanese government MIS projects. All features are size/function-count
    metrics; MM (man-months) is the effort target.

    KLOC  <- features x complexity x 1.5 (same proxy as kemerer)
    SCRN  <- num_features (one screen per feature/use-case)
    FORM  <- num_features x 0.5 (forms ≈ half of screens)
    FILE  <- num_features x 0.3 (master files ≈ 30% of features)
    ESCRN <- SCRN x 8  (equivalent count after complexity adjustment)
    EFORM <- FORM x 8
    EFILE <- FILE x 8
    """
    complexity = _COMPLEXITY_SCORE[pm.feature_complexity]
    kloc = pm.num_features * complexity * 1.5
    scrn = float(pm.num_features)
    form = pm.num_features * 0.5
    file_ = pm.num_features * 0.3
    return [
        kloc,        # KLOC
        scrn,        # SCRN
        form,        # FORM
        file_,       # FILE
        scrn * 8.0,  # ESCRN
        form * 8.0,  # EFORM
        file_ * 8.0, # EFILE
    ]


_FEATURE_BUILDERS: dict[str, callable] = {
    "desharnais": _features_desharnais,
    "china": _features_china,
    "kitchenham": _features_kitchenham,
    "subbiah": _features_subbiah,
    "isbsg10": _features_isbsg10,
    "kemerer": _features_kemerer,
    "nasa93": _features_nasa93,
    "miyazaki94": _features_miyazaki94,
}

SUPPORTED_STEMS: frozenset[str] = frozenset(_FEATURE_BUILDERS.keys())


# ── Public API ────────────────────────────────────────────────────────────────

def estimate(pm_input: PMInput, models_dir: str | None = None) -> BridgeResult | None:
    """Map PM inputs to ML features and return a BridgeResult.

    Returns None if no trained models are found or prediction fails.
    """
    mdir = models_dir or os.path.normpath(_MODELS_DIR)
    best = select_best_estimator(mdir, supported_stems=SUPPORTED_STEMS, min_pred25=CONFIDENCE_THRESHOLD)
    if best is None:
        return None

    builder = _FEATURE_BUILDERS.get(best.stem)
    if builder is None:
        return None

    raw_features = builder(pm_input)
    X = np.array([raw_features], dtype=float)

    loaded = load_estimator_models(best.stem, mdir)
    if loaded.scaler is not None:
        X_scaled = loaded.scaler.transform(X)
    else:
        X_scaled = X

    if best.model_name == "Hybrid":
        effort_pm = run_hybrid_estimate(X_scaled, loaded, best.effort_unit)
    else:
        effort_pm = run_estimate(best.model_name, X_scaled, loaded, best.effort_unit)

    if effort_pm is None:
        return None

    return BridgeResult(effort_pm=effort_pm, model_info=best)


def simulate(base_input: PMInput, field: str, value, models_dir: str | None = None) -> BridgeResult | None:
    """Return a new BridgeResult with one PMInput field overridden.

    All other fields are preserved from base_input.
    """
    overridden = copy.copy(base_input)
    object.__setattr__(overridden, field, value)
    overridden._validate()
    return estimate(overridden, models_dir)


def compare(base: BridgeResult, simulated: BridgeResult) -> dict:
    """Compare two BridgeResults and return delta statistics.

    Returns:
        delta_effort : absolute difference in person-months (simulated - base)
        delta_pct    : percentage change
        direction    : 'increase' | 'decrease' | 'unchanged'
    """
    delta = simulated.effort_pm - base.effort_pm
    if base.effort_pm == 0:
        delta_pct = 0.0
    else:
        delta_pct = (delta / base.effort_pm) * 100.0

    if abs(delta_pct) < 0.001:
        direction = "unchanged"
    elif delta > 0:
        direction = "increase"
    else:
        direction = "decrease"

    return {
        "delta_effort": round(delta, 4),
        "delta_pct": round(delta_pct, 2),
        "direction": direction,
    }
