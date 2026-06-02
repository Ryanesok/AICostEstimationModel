"""Step-by-step wizard UI for collecting PM project parameters.

Five steps: Technical -> Team -> Business -> Risk -> Cost & Budget.
Categorical fields use CTkOptionMenu (dropdown).
Numeric fields use CTkSlider with a live value label.
On completion the on_complete callback is invoked with (PMInput, salary_per_month, initial_budget).
"""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from core.schema import PMInput

# ── Step definitions ──────────────────────────────────────────────────────────
# Each field entry: (field_name, label, widget_type, options_or_range)
#   widget_type = 'dropdown' | 'slider'
#   options_or_range:
#     dropdown -> list of str options
#     slider   -> (min, max, step, default)

_STEPS: list[tuple[str, list[tuple]]] = [
    ("Technical Parameters", [
        ("num_features",           "Number of Features",          "slider",   (1, 100, 1, 10)),
        ("feature_complexity",     "Feature Complexity",          "dropdown", ["low", "medium", "high"]),
        ("num_apis",               "Number of API Endpoints",     "slider",   (0, 50, 1, 5)),
        ("platform",               "Target Platform",             "dropdown", ["web", "mobile", "desktop", "embedded", "other"]),
        ("third_party_integrations","Third-party Integrations",   "slider",   (0, 20, 1, 2)),
        ("security_level",         "Security Level",              "dropdown", ["basic", "standard", "high"]),
    ]),
    ("Team Parameters", [
        ("num_developers",  "Number of Developers",   "slider",   (1, 30, 1, 4)),
        ("seniority",       "Team Seniority",         "dropdown", ["junior", "mid", "senior", "mixed"]),
        ("stack_experience","Stack Experience",        "dropdown", ["low", "medium", "high"]),
    ]),
    ("Business Parameters", [
        ("deadline_months",          "Deadline (months)",         "slider",   (1, 60, 1, 6)),
        ("budget_constraint",        "Budget Constraint",         "dropdown", ["none", "soft", "hard"]),
        ("requirement_change_risk",  "Requirement Change Risk",   "dropdown", ["low", "medium", "high"]),
    ]),
    ("Risk Parameters", [
        ("uncertainty_level",    "Uncertainty Level",      "dropdown", ["low", "medium", "high"]),
        ("technical_debt",       "Technical Debt",         "dropdown", ["none", "low", "high"]),
        ("external_dependencies","External Dependencies",  "slider",   (0, 20, 1, 1)),
    ]),
    ("Cost & Budget", [
        ("salary_per_month", "Salary per Developer/Month (IDR)", "slider", (500_000, 20_000_000, 500_000, 5_000_000)),
        ("initial_budget",   "Initial Budget (IDR)",             "slider", (10_000_000, 2_000_000_000, 10_000_000, 200_000_000)),
    ]),
]

_STEP_NAMES = [s[0] for s in _STEPS]

# Currency formatter for large IDR slider fields
_SLIDER_FORMATTERS: dict[str, Callable[[float], str]] = {
    "salary_per_month": lambda v: f"Rp {int(v):,}",
    "initial_budget":   lambda v: f"Rp {int(v):,}",
}


class WizardApp(ctk.CTk):
    """Main wizard window. Calls on_complete(PMInput, salary_per_month, initial_budget) on submit."""

    def __init__(self, on_complete: Callable):
        super().__init__()
        self.title("Software Effort Estimation -- Project Setup")
        self.geometry("720x600")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._on_complete = on_complete
        self._current_step = 0
        # Persisted widget values per field
        self._values: dict[str, ctk.Variable] = {}
        self._init_variables()

        # ── Layout ────────────────────────────────────────────────────────────
        self._header = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=18, weight="bold"))
        self._header.pack(pady=(20, 4))

        self._step_indicator = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12), text_color="gray")
        self._step_indicator.pack(pady=(0, 12))

        self._content_frame = ctk.CTkScrollableFrame(self, width=660, height=420)
        self._content_frame.pack(padx=20, pady=4)

        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(pady=12)
        self._back_btn = ctk.CTkButton(nav, text="Back", width=120, command=self._go_back)
        self._back_btn.grid(row=0, column=0, padx=8)
        self._next_btn = ctk.CTkButton(nav, text="Next", width=120, command=self._go_next)
        self._next_btn.grid(row=0, column=1, padx=8)

        self._render_step()

    # ── Variable initialisation ───────────────────────────────────────────────

    def _init_variables(self) -> None:
        for _, fields in _STEPS:
            for field_name, _, widget_type, opts in fields:
                if widget_type == "dropdown":
                    var = ctk.StringVar(value=opts[0])
                else:
                    _, _, _, default = opts
                    var = ctk.DoubleVar(value=float(default))
                self._values[field_name] = var

    # ── Step rendering ────────────────────────────────────────────────────────

    def _render_step(self) -> None:
        step_title, fields = _STEPS[self._current_step]
        self._header.configure(text=step_title)
        self._step_indicator.configure(
            text=f"Step {self._current_step + 1} of {len(_STEPS)}"
        )
        self._back_btn.configure(state="normal" if self._current_step > 0 else "disabled")
        last_step = self._current_step == len(_STEPS) - 1
        self._next_btn.configure(text="Estimate" if last_step else "Next")

        # Clear content frame
        for widget in self._content_frame.winfo_children():
            widget.destroy()

        for field_name, label, widget_type, opts in fields:
            row = ctk.CTkFrame(self._content_frame, fg_color="transparent")
            row.pack(fill="x", pady=6)

            lbl = ctk.CTkLabel(row, text=label, width=220, anchor="w")
            lbl.pack(side="left", padx=(8, 12))

            var = self._values[field_name]

            if widget_type == "dropdown":
                ctk.CTkOptionMenu(row, values=opts, variable=var, width=200).pack(side="left")
            else:
                min_val, max_val, step, _ = opts
                formatter = _SLIDER_FORMATTERS.get(field_name, lambda v: str(int(v)))
                lbl_width = 140 if field_name in _SLIDER_FORMATTERS else 50
                value_label = ctk.CTkLabel(row, text=formatter(var.get()), width=lbl_width, anchor="w")

                def _make_cmd(vl=value_label, v=var, st=step, fmt=formatter):
                    def _cmd(val):
                        rounded = round(float(val) / st) * st
                        v.set(rounded)
                        vl.configure(text=fmt(rounded))
                    return _cmd

                slider = ctk.CTkSlider(
                    row, from_=min_val, to=max_val, number_of_steps=int((max_val - min_val) / step),
                    variable=var, command=_make_cmd(),
                )
                slider.pack(side="left", padx=4)
                value_label.pack(side="left", padx=4)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _go_back(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._render_step()

    def _go_next(self) -> None:
        if self._current_step < len(_STEPS) - 1:
            self._current_step += 1
            self._render_step()
        else:
            self._submit()

    def _submit(self) -> None:
        v = self._values
        pm = PMInput(
            num_features=int(v["num_features"].get()),
            feature_complexity=v["feature_complexity"].get(),
            num_apis=int(v["num_apis"].get()),
            platform=v["platform"].get(),
            third_party_integrations=int(v["third_party_integrations"].get()),
            security_level=v["security_level"].get(),
            num_developers=int(v["num_developers"].get()),
            seniority=v["seniority"].get(),
            stack_experience=v["stack_experience"].get(),
            deadline_months=float(v["deadline_months"].get()),
            budget_constraint=v["budget_constraint"].get(),
            requirement_change_risk=v["requirement_change_risk"].get(),
            uncertainty_level=v["uncertainty_level"].get(),
            technical_debt=v["technical_debt"].get(),
            external_dependencies=int(v["external_dependencies"].get()),
        )
        salary = float(v["salary_per_month"].get())
        budget = float(v["initial_budget"].get())
        self._on_complete(pm, salary, budget)
