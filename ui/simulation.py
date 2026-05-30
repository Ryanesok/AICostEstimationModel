"""Simulation panel widget for what-if scenario analysis.

Allows the PM to pick one PMInput field, enter a new value,
click Recalculate, and see the updated effort alongside the
delta vs the base estimate.
"""
from __future__ import annotations

import dataclasses

import customtkinter as ctk

from core.schema import PMInput
from core import estimator_bridge
from core.estimator_bridge import BridgeResult

# Human-readable labels and input specs for each PMInput field
# Each entry: (display_label, widget_type, options_or_range)
#   widget_type = 'dropdown' | 'slider'
_FIELD_CONFIG: dict[str, tuple[str, str, list | tuple]] = {
    "num_features":           ("Number of Features",         "slider",   (1, 100, 1)),
    "feature_complexity":     ("Feature Complexity",         "dropdown", ["low", "medium", "high"]),
    "num_apis":               ("Number of API Endpoints",    "slider",   (0, 50, 1)),
    "platform":               ("Target Platform",            "dropdown", ["web", "mobile", "desktop", "embedded", "other"]),
    "third_party_integrations":("Third-party Integrations",  "slider",   (0, 20, 1)),
    "security_level":         ("Security Level",             "dropdown", ["basic", "standard", "high"]),
    "num_developers":         ("Number of Developers",       "slider",   (1, 30, 1)),
    "seniority":              ("Team Seniority",             "dropdown", ["junior", "mid", "senior", "mixed"]),
    "stack_experience":       ("Stack Experience",           "dropdown", ["low", "medium", "high"]),
    "deadline_months":        ("Deadline (months)",          "slider",   (1, 60, 1)),
    "budget_constraint":      ("Budget Constraint",          "dropdown", ["none", "soft", "hard"]),
    "requirement_change_risk":("Requirement Change Risk",    "dropdown", ["low", "medium", "high"]),
    "uncertainty_level":      ("Uncertainty Level",          "dropdown", ["low", "medium", "high"]),
    "technical_debt":         ("Technical Debt",             "dropdown", ["none", "low", "high"]),
    "external_dependencies":  ("External Dependencies",      "slider",   (0, 20, 1)),
}


class SimulationPanel(ctk.CTkFrame):
    """What-if simulation widget.

    Parameters
    ----------
    parent       : parent CTk widget
    base_input   : the original PMInput used for the base estimate
    base_result  : BridgeResult for the base estimate
    """

    def __init__(self, parent, base_input: PMInput, base_result: BridgeResult):
        super().__init__(parent)
        self._base_input = base_input
        self._base_result = base_result
        self._sim_value_var: ctk.Variable | None = None
        self._sim_widget: ctk.CTkWidget | None = None

        # ── Header ────────────────────────────────────────────────────────────
        ctk.CTkLabel(self, text="Scenario Simulation",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12, 4))

        ctk.CTkLabel(
            self,
            text=f"Base estimate: {base_result.effort_pm:.1f} person-months",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa",
        ).pack(pady=(0, 8))

        # ── Field selector ────────────────────────────────────────────────────
        sel_row = ctk.CTkFrame(self, fg_color="transparent")
        sel_row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(sel_row, text="Change field:", width=120, anchor="w").pack(side="left")

        field_names = list(_FIELD_CONFIG.keys())
        self._field_var = ctk.StringVar(value=field_names[0])
        self._field_selector = ctk.CTkOptionMenu(
            sel_row, values=field_names,
            variable=self._field_var,
            command=self._on_field_change,
            width=220,
        )
        self._field_selector.pack(side="left", padx=8)

        # ── Value input area ──────────────────────────────────────────────────
        self._value_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._value_frame.pack(fill="x", padx=16, pady=4)

        # ── Recalculate button ────────────────────────────────────────────────
        ctk.CTkButton(self, text="Recalculate", command=self._recalculate).pack(pady=8)

        # ── Result display ────────────────────────────────────────────────────
        self._result_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12))
        self._result_label.pack(pady=4)
        self._delta_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11), text_color="#aaaaaa")
        self._delta_label.pack(pady=2)

        # Render initial value control for the first field
        self._on_field_change(field_names[0])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _on_field_change(self, field_name: str) -> None:
        for w in self._value_frame.winfo_children():
            w.destroy()

        label_text, widget_type, opts = _FIELD_CONFIG[field_name]
        current_val = getattr(self._base_input, field_name)

        ctk.CTkLabel(self._value_frame, text=f"New value for '{label_text}':",
                     width=220, anchor="w").pack(side="left", padx=(0, 8))

        if widget_type == "dropdown":
            self._sim_value_var = ctk.StringVar(value=str(current_val))
            self._sim_widget = ctk.CTkOptionMenu(
                self._value_frame, values=opts,
                variable=self._sim_value_var, width=180,
            )
        else:
            min_v, max_v, step = opts
            self._sim_value_var = ctk.DoubleVar(value=float(current_val))
            val_label = ctk.CTkLabel(self._value_frame, text=str(int(float(current_val))), width=40)

            def _cmd(val, vl=val_label, var=self._sim_value_var, st=step):
                rounded = round(float(val) / st) * st
                var.set(rounded)
                vl.configure(text=str(int(rounded)))

            self._sim_widget = ctk.CTkSlider(
                self._value_frame, from_=min_v, to=max_v,
                number_of_steps=int((max_v - min_v) / step),
                variable=self._sim_value_var, command=_cmd,
            )
            self._sim_widget.pack(side="left")
            val_label.pack(side="left", padx=4)
            return

        self._sim_widget.pack(side="left")

    def _recalculate(self) -> None:
        field_name = self._field_var.get()
        raw = self._sim_value_var.get()

        # Coerce to the correct Python type
        field_type = type(getattr(self._base_input, field_name))
        try:
            value = field_type(raw) if field_type != float else float(raw)
            if field_type == int:
                value = int(float(raw))
        except (ValueError, TypeError):
            self._result_label.configure(text="Invalid value.", text_color="#E74C3C")
            return

        try:
            sim_result = estimator_bridge.simulate(self._base_input, field_name, value)
        except ValueError as exc:
            self._result_label.configure(text=f"Validation error: {exc}", text_color="#E74C3C")
            return

        if sim_result is None:
            self._result_label.configure(text="Estimation failed.", text_color="#E74C3C")
            return

        cmp = estimator_bridge.compare(self._base_result, sim_result)
        self._result_label.configure(
            text=f"Simulated effort: {sim_result.effort_pm:.1f} person-months",
            text_color="white",
        )
        direction_arrow = {"increase": "+", "decrease": "-", "unchanged": ""}[cmp["direction"]]
        color = {"increase": "#E74C3C", "decrease": "#27AE60", "unchanged": "#aaaaaa"}[cmp["direction"]]
        self._delta_label.configure(
            text=f"Delta: {direction_arrow}{cmp['delta_effort']:.2f} pm  ({direction_arrow}{cmp['delta_pct']:.1f}%)",
            text_color=color,
        )
