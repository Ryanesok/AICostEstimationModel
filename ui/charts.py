"""Chart rendering functions for the estimation results UI.

All functions embed a matplotlib figure into a given CTk parent frame,
replacing any previously embedded chart in that frame.

Available charts:
  render_dominance_pie  - effort share per PM dimension (pie)
  render_cost_bar       - effort breakdown bar chart
  render_risk_heatmap   - 2D heatmap of risk-related PM fields
  render_timeline       - Gantt-style project timeline estimate
"""
from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.schema import PMInput
from core.estimator_bridge import BridgeResult

_DIMENSION_COLORS = {
    "Technical": "#4A90D9",
    "Team":      "#27AE60",
    "Business":  "#E67E22",
    "Risk":      "#E74C3C",
}


def _clear_frame(frame) -> None:
    for widget in frame.winfo_children():
        widget.destroy()


def _embed(fig: plt.Figure, frame) -> None:
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)


# ── Public rendering functions ────────────────────────────────────────────────

def render_dominance_pie(frame, dominance_dict: dict) -> None:
    """Embed a pie chart of effort share per PM dimension into frame."""
    _clear_frame(frame)

    labels = list(dominance_dict.keys())
    sizes = [dominance_dict[k] for k in labels]
    colors = [_DIMENSION_COLORS.get(k, "#95A5A6") for k in labels]

    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=140,
        textprops={"color": "white", "fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Effort Dominance", color="white", fontsize=11, pad=8)
    fig.tight_layout()
    _embed(fig, frame)


def render_cost_bar(frame, result: BridgeResult, dominance_dict: dict) -> None:
    """Embed a horizontal bar chart of effort per dimension into frame."""
    _clear_frame(frame)

    dims = list(dominance_dict.keys())
    shares = [dominance_dict[d] for d in dims]
    effort_pm = result.effort_pm
    values = [s / 100.0 * effort_pm for s in shares]
    colors = [_DIMENSION_COLORS.get(d, "#95A5A6") for d in dims]

    fig, ax = plt.subplots(figsize=(4.5, 2.8), facecolor="#1e1e2e")
    ax.set_facecolor("#2a2a3e")
    bars = ax.barh(dims, values, color=colors, height=0.5)
    ax.set_xlabel("Person-months", color="white", fontsize=9)
    ax.set_title("Effort by Dimension", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    for bar, val in zip(bars, values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", color="white", fontsize=8)
    fig.tight_layout()
    _embed(fig, frame)


def render_risk_heatmap(frame, pm_input: PMInput) -> None:
    """Embed a risk heatmap derived from risk-related PM fields into frame."""
    _clear_frame(frame)

    _ord = {"none": 0, "low": 1, "medium": 2, "high": 3,
            "basic": 0, "standard": 1, "soft": 1, "hard": 2}

    rows = ["Uncertainty", "Tech Debt", "Ext Deps", "Security", "Req Change"]
    values = [
        _ord.get(pm_input.uncertainty_level, 1),
        _ord.get(pm_input.technical_debt, 0),
        min(pm_input.external_dependencies, 3),
        _ord.get(pm_input.security_level, 1),
        _ord.get(pm_input.requirement_change_risk, 1),
    ]

    data = np.array(values).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=3, aspect="auto")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, color="white", fontsize=8)
    ax.set_xticks([])
    ax.set_title("Risk Profile", color="white", fontsize=11)

    for i, val in enumerate(values):
        label = ["Low", "Low", "Med", "High"][min(val, 3)]
        ax.text(0, i, label, ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.08).ax.tick_params(colors="white")
    fig.tight_layout()
    _embed(fig, frame)


def render_timeline(frame, pm_input: PMInput, result: BridgeResult) -> None:
    """Embed a Gantt-style timeline estimate into frame.

    Phases are rough proportions based on typical software project breakdown.
    """
    _clear_frame(frame)

    effort = result.effort_pm
    deadline = pm_input.deadline_months

    # Phase proportions (person-months and calendar months)
    phases = [
        ("Requirements",   0.10, 0.12),
        ("Design",         0.12, 0.15),
        ("Development",    0.50, 0.55),
        ("Testing / QA",   0.18, 0.22),
        ("Deployment",     0.10, 0.08),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.0), facecolor="#1e1e2e")
    ax.set_facecolor("#2a2a3e")

    palette = ["#4A90D9", "#27AE60", "#8E44AD", "#E67E22", "#E74C3C"]
    start = 0.0
    for i, (name, effort_frac, time_frac) in enumerate(phases):
        duration = deadline * time_frac
        ax.barh(name, duration, left=start, color=palette[i % len(palette)], height=0.5)
        ax.text(start + duration / 2, name,
                f"{effort * effort_frac:.1f} pm",
                ha="center", va="center", color="white", fontsize=7)
        start += duration

    ax.set_xlabel("Months", color="white", fontsize=9)
    ax.set_xlim(0, deadline * 1.05)
    ax.set_title(f"Timeline ({deadline:.0f} month project, {effort:.1f} pm total)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    fig.tight_layout()
    _embed(fig, frame)
