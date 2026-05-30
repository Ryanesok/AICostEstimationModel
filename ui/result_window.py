"""Result window: full estimation result view opened after the wizard completes.

Shows:
- Effort estimate + confidence range + cost summary (total cost, surplus/deficit)
- Model accuracy badge (quality label + warn flag)
- Tabbed chart area: Dominance | Cost | Risk | Timeline | History
- Scrollable explainer text
- Simulation panel (shown on button click)
- Decision support panel: overrun risk + optimization suggestions + risk advice
"""
from __future__ import annotations

import customtkinter as ctk

from core.schema import PMInput
from core.estimator_bridge import BridgeResult
from core import (
    confidence as conf_mod,
    dominance as dom_mod,
    cost_calculator,
    accuracy_reporter,
    decision_support,
    past_project,
)
from explainer import explain
from ui import charts
from ui.simulation import SimulationPanel

_RISK_COLORS = {"low": "#27AE60", "medium": "#E67E22", "high": "#E74C3C"}
_QUALITY_COLORS = {"Excellent": "#27AE60", "Good": "#4A90D9", "Fair": "#E67E22", "Unknown": "#888"}


class ResultWindow(ctk.CTkToplevel):
    """Full result display window opened after wizard completes."""

    def __init__(
        self,
        parent,
        pm_input: PMInput,
        result: BridgeResult,
        salary_per_month: float,
        initial_budget: float,
    ):
        super().__init__(parent)
        self.title("Estimation Result")
        self.geometry("900x720")
        self.resizable(True, True)

        self._pm = pm_input
        self._result = result
        self._salary = salary_per_month
        self._budget = initial_budget

        # Compute all outputs
        self._ci = conf_mod.compute(result)
        self._dom = dom_mod.compute(pm_input, result)
        self._acc = accuracy_reporter.summarize(result)
        self._cost = cost_calculator.compute(
            result.effort_pm, salary_per_month, pm_input.deadline_months, pm_input.num_developers
        )
        self._explanation = explain(pm_input, result, self._dom, self._ci)

        # Save to history first, then find similar (excludes the record just saved)
        past_project.save(pm_input, result, self._ci)
        self._similar = past_project.find_similar(pm_input, top_n=3)

        self._build_ui()

    def _build_ui(self) -> None:
        # ── Header strip ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=0)
        header.pack(fill="x", pady=(0, 0))

        effort = self._result.effort_pm
        lower = self._ci["lower"]
        upper = self._ci["upper"]
        conf_pct = self._ci["confidence_pct"]

        ctk.CTkLabel(
            header,
            text=f"{effort:.1f}  person-months",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkLabel(
            header,
            text=f"Range: {lower:.1f} - {upper:.1f} pm  |  Confidence: {conf_pct:.0f}%",
            font=ctk.CTkFont(size=12),
            text_color="#aaaaaa",
        ).pack(side="left", padx=8)

        # Model quality badge
        q_label = self._acc["quality_label"]
        q_color = _QUALITY_COLORS.get(q_label, "#888")
        ctk.CTkLabel(
            header,
            text=f"Model: {q_label}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=q_color,
        ).pack(side="right", padx=20)

        # ── Cost summary strip ─────────────────────────────────────────────────
        self._build_cost_strip()

        # Warn banner if model quality is poor
        if self._acc["warn"]:
            warn_bar = ctk.CTkFrame(self, fg_color="#5a2020", corner_radius=0)
            warn_bar.pack(fill="x")
            ctk.CTkLabel(
                warn_bar,
                text=f"  Warning: {self._acc['warn_message']}",
                font=ctk.CTkFont(size=10),
                text_color="#ffaaaa",
                anchor="w",
            ).pack(fill="x", padx=8, pady=3)

        # ── Main body: tabs + explainer ───────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=4)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Left: tabview for charts + history
        tab_view = ctk.CTkTabview(body, width=480)
        tab_view.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        for tab_name in ("Dominance", "Cost", "Risk", "Timeline", "History"):
            tab_view.add(tab_name)

        charts.render_dominance_pie(tab_view.tab("Dominance"), self._dom)
        charts.render_cost_bar(tab_view.tab("Cost"), self._result, self._dom)
        charts.render_risk_heatmap(tab_view.tab("Risk"), self._pm)
        charts.render_timeline(tab_view.tab("Timeline"), self._pm, self._result)
        self._build_history_tab(tab_view.tab("History"))

        # Right: explainer text
        right = ctk.CTkFrame(body)
        right.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(right, text="Explanation", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(8, 2))
        txt = ctk.CTkTextbox(right, wrap="word", font=ctk.CTkFont(size=10))
        txt.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        txt.insert("end", self._explanation)
        txt.configure(state="disabled")

        # ── Decision support strip ────────────────────────────────────────────
        self._build_decision_strip()

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=6)
        ctk.CTkButton(btn_row, text="Simulate", width=120, command=self._open_simulation).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Close", width=120, fg_color="#555", command=self.destroy).pack(side="right", padx=4)

    def _build_cost_strip(self) -> None:
        """Secondary strip showing total cost, team sizing, and budget surplus/deficit."""
        surplus = self._budget - self._cost.total_cost
        surplus_color = "#27AE60" if surplus >= 0 else "#E74C3C"
        surplus_label = f"Surplus: {self._cost.currency} {abs(surplus):,.0f}" if surplus >= 0 \
                        else f"Deficit: {self._cost.currency} {abs(surplus):,.0f}"

        strip = ctk.CTkFrame(self, fg_color="#12122a", corner_radius=0)
        strip.pack(fill="x")

        ctk.CTkLabel(
            strip,
            text=f"Total Cost: {self._cost.formatted_total}",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=20, pady=6)

        ctk.CTkLabel(
            strip,
            text=f"Team Needed: {self._cost.team_needed} devs",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa",
        ).pack(side="left", padx=12)

        ctk.CTkLabel(
            strip,
            text=f"Budget: {self._cost.currency} {self._budget:,.0f}",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa",
        ).pack(side="left", padx=12)

        ctk.CTkLabel(
            strip,
            text=surplus_label,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=surplus_color,
        ).pack(side="left", padx=20)

    def _build_history_tab(self, frame: ctk.CTkFrame) -> None:
        """Render similar past projects comparison, or first-estimation message."""
        if not self._similar:
            ctk.CTkLabel(
                frame,
                text="First estimation -- no past projects to compare.",
                font=ctk.CTkFont(size=12),
                text_color="#aaaaaa",
            ).pack(expand=True)
            return

        # Header row
        header = ctk.CTkFrame(frame, fg_color="#2a2a3e")
        header.pack(fill="x", padx=4, pady=(8, 2))
        for col, w in [("Date", 120), ("Effort (pm)", 90), ("Confidence", 90), ("Similarity", 80), ("Dataset", 90)]:
            ctk.CTkLabel(header, text=col, font=ctk.CTkFont(size=10, weight="bold"), width=w, anchor="w").pack(side="left", padx=4)

        scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        current_effort = self._result.effort_pm
        for rec in self._similar:
            past_effort = rec.get("effort_pm", 0)
            delta = past_effort - current_effort
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.1f} pm)"
            sim_pct = round(rec.get("similarity", 0) * 100, 1)
            ts = rec.get("timestamp", "")[:10]
            conf = rec.get("confidence_pct")
            conf_str = f"{conf:.0f}%" if conf is not None else "n/a"
            dataset = rec.get("dataset", "?")

            row = ctk.CTkFrame(scroll, fg_color="#1e1e2e")
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(row, text=ts,           width=120, font=ctk.CTkFont(size=10), anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=f"{past_effort:.1f}{delta_str}", width=90,  font=ctk.CTkFont(size=10), anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=conf_str,     width=90,  font=ctk.CTkFont(size=10), anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=f"{sim_pct}%",width=80,  font=ctk.CTkFont(size=10), anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=dataset,      width=90,  font=ctk.CTkFont(size=10), anchor="w").pack(side="left", padx=4)

    def _build_decision_strip(self) -> None:
        strip = ctk.CTkFrame(self, fg_color="#1e1e2e")
        strip.pack(fill="x", padx=12, pady=4)
        strip.columnconfigure((0, 1, 2), weight=1)

        # Overrun risk (using initial_budget as budget proxy in person-months)
        budget_pm = self._budget / max(self._salary, 1)
        overrun = decision_support.detect_overrun_risk(self._pm, self._result, budget_pm)
        risk_color = _RISK_COLORS[overrun["risk_level"]]

        risk_frame = ctk.CTkFrame(strip, fg_color="transparent")
        risk_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(risk_frame, text="Overrun Risk", font=ctk.CTkFont(size=11, weight="bold")).pack()
        ctk.CTkLabel(
            risk_frame,
            text=overrun["risk_level"].upper(),
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=risk_color,
        ).pack()
        ctk.CTkLabel(
            risk_frame,
            text=overrun["message"][:70] + ("..." if len(overrun["message"]) > 70 else ""),
            font=ctk.CTkFont(size=9),
            text_color="#aaaaaa",
            wraplength=220,
        ).pack()

        # Top optimization suggestion
        opts = decision_support.suggest_optimizations(self._pm, self._result, budget_pm)
        opt_frame = ctk.CTkFrame(strip, fg_color="transparent")
        opt_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(opt_frame, text="Top Optimization", font=ctk.CTkFont(size=11, weight="bold")).pack()
        if opts:
            ctk.CTkLabel(
                opt_frame,
                text=opts[0]["description"],
                font=ctk.CTkFont(size=9),
                text_color="#aaaaaa",
                wraplength=220,
            ).pack(pady=4)
        else:
            ctk.CTkLabel(opt_frame, text="Within budget -- no changes needed.", font=ctk.CTkFont(size=9), text_color="#27AE60").pack(pady=4)

        # Top risk advice
        advice = decision_support.advise_risks(self._pm)
        adv_frame = ctk.CTkFrame(strip, fg_color="transparent")
        adv_frame.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(adv_frame, text="Risk Advice", font=ctk.CTkFont(size=11, weight="bold")).pack()
        if advice:
            ctk.CTkLabel(
                adv_frame,
                text=advice[0]["advice"][:120],
                font=ctk.CTkFont(size=9),
                text_color="#aaaaaa",
                wraplength=220,
            ).pack(pady=4)
        else:
            ctk.CTkLabel(adv_frame, text="No high-risk factors detected.", font=ctk.CTkFont(size=9), text_color="#27AE60").pack(pady=4)

    def _open_simulation(self) -> None:
        sim_win = ctk.CTkToplevel(self)
        sim_win.title("Scenario Simulation")
        sim_win.geometry("560x420")
        panel = SimulationPanel(sim_win, self._pm, self._result)
        panel.pack(fill="both", expand=True)
