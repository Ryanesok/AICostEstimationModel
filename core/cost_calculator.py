"""Convert effort estimate (person-months) to project cost and team sizing."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CostResult:
    total_cost: float
    team_needed: int
    actual_duration_months: float
    cost_per_developer: float
    currency: str
    formatted_total: str


def _format_idr(amount: float) -> str:
    return f"Rp {amount:,.0f}"


def _format_usd(amount: float) -> str:
    return f"${amount:,.0f}"


_FORMATTERS = {"IDR": _format_idr, "USD": _format_usd}


def compute(
    effort_pm: float,
    salary_per_month: float,
    deadline_months: float,
    num_developers: int,
    currency: str = "IDR",
) -> CostResult:
    """Compute total project cost and team sizing from effort estimate.

    Parameters
    ----------
    effort_pm        : estimated effort in person-months
    salary_per_month : average monthly salary per developer
    deadline_months  : project duration in months (used for team_needed)
    num_developers   : planned team size (used for cost_per_developer)
    currency         : 'IDR' or 'USD' for formatting
    """
    total_cost = effort_pm * salary_per_month
    team_needed = math.ceil(effort_pm / max(deadline_months, 0.1))
    actual_duration = effort_pm / max(num_developers, 1)
    cost_per_dev = total_cost / max(num_developers, 1)

    formatter = _FORMATTERS.get(currency.upper(), _format_idr)
    formatted = formatter(total_cost)

    return CostResult(
        total_cost=total_cost,
        team_needed=team_needed,
        actual_duration_months=actual_duration,
        cost_per_developer=cost_per_dev,
        currency=currency.upper(),
        formatted_total=formatted,
    )
