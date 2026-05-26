import csv
import datetime
import json
import math
import os
from tkinter import filedialog

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import yaml

from pipeline.estimator import (
    EstimatorResult,
    LoadedModels,
    load_estimator_models,
    run_estimate,
    run_hybrid_estimate,
    select_best_estimator,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODELS_DIR = "models"
CONFIG_FILE = "pipeline/dataset_config.yaml"
HOURS_PER_MONTH = 160

_COMPARISON_MODELS = ["SVR", "Linear Regression", "Random Forest", "XGBoost", "LSTM", "Hybrid"]
_METRIC_KEYS: dict[str, dict[str, str | None]] = {
    "SVR":              {"mae": "cv_mae_mean",      "rmse": "cv_rmse_mean",      "r2": "cv_r2_mean"},
    "Linear Regression":{"mae": "lr_cv_mae_mean",   "rmse": "lr_cv_rmse_mean",   "r2": "lr_cv_r2_mean"},
    "Random Forest":    {"mae": "rf_cv_mae_mean",    "rmse": "rf_cv_rmse_mean",   "r2": "rf_cv_r2_mean"},
    "XGBoost":          {"mae": "xgb_cv_mae_mean",   "rmse": "xgb_cv_rmse_mean",  "r2": "xgb_cv_r2_mean"},
    "LSTM":             {"mae": "lstm_cv_mae_mean",  "rmse": "lstm_cv_rmse_mean", "r2": "lstm_cv_r2_mean"},
    "Hybrid":           {"mae": None,               "rmse": None,                "r2": None},
}


# ── Config helpers ────────────────────────────────────────────────────────────

def load_field_config_from_yaml(stem: str) -> list[tuple] | None:
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get("datasets", {}).get(stem.lower())
    if cfg is None:
        return None
    features = cfg.get("features", [])
    labels = cfg.get("labels", {})
    hints = cfg.get("hints", {})
    return [(labels.get(col, col), col, hints.get(col, "")) for col in features]


def load_slider_config_from_yaml(stem: str) -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get("datasets", {}).get(stem.lower(), {})
    return cfg.get("sliders", {})


def _default_fields(n: int) -> list[tuple]:
    return [(f"Feature {i + 1}", f"f{i}", f"Input fitur ke-{i + 1}") for i in range(n)]


def _quality_info(mmre: float | None) -> tuple[str, str]:
    """Returns (label_text, color) from MMRE value."""
    if mmre is None:
        return "Akurasi tidak tersedia", "gray60"
    if mmre <= 0.25:
        return "Akurasi Tinggi  ✓", "#98c379"
    if mmre <= 0.50:
        return "Cukup Baik", "#e5c07b"
    return "Perlu Verifikasi Manual", "#e06c75"


# ── Input persistence ─────────────────────────────────────────────────────────

LAST_INPUTS_DIR = "last_inputs"


def save_last_inputs(dataset_name: str, field_values: dict) -> None:
    os.makedirs(LAST_INPUTS_DIR, exist_ok=True)
    path = os.path.join(LAST_INPUTS_DIR, f"{dataset_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(field_values, f)


def load_last_inputs(dataset_name: str) -> dict:
    path = os.path.join(LAST_INPUTS_DIR, f"{dataset_name}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


# ── App ───────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Cost Estimator")
        self.geometry("1100x820")
        self.resizable(False, False)

        self._best: EstimatorResult | None = select_best_estimator(MODELS_DIR)
        self._models: LoadedModels | None = None
        self._current_estimate: float | None = None
        self._field_readers: dict[str, callable] = {}
        self._field_resetters: dict[str, callable] = {}
        self._entries: dict[str, ctk.CTkEntry] = {}
        self._detail_visible: bool = False
        self._last_result: dict | None = None
        self._model_check_vars: dict[str, ctk.BooleanVar] = {}
        self._metrics_cache: dict = {}

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=420)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(self, corner_radius=12, width=400)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        left.grid_columnconfigure(1, weight=1)
        self._left = left

        ctk.CTkLabel(
            left, text="Input Fitur",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(14, 4), padx=20)

        if self._best is None:
            self._build_error_banner(left)
        else:
            self._models = load_estimator_models(self._best.stem, MODELS_DIR)
            self._build_form_area(left)

        right = ctk.CTkScrollableFrame(self, corner_radius=12)
        right.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        self._right = right
        self._build_results_panel(right)

    def _build_error_banner(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(
            frame, text="Model Tidak Tersedia",
            text_color="#e06c75", font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=(14, 6))
        ctk.CTkLabel(
            frame,
            text=(
                "Tidak ada model terlatih di folder 'models/'.\n\n"
                "Jalankan langkah berikut:\n"
                "  1. python pipeline/downloader.py\n"
                "  2. python pipeline/build.py --configure\n"
                "  3. python pipeline/build.py --train"
            ),
            text_color="gray70", font=ctk.CTkFont(size=12), justify="left",
        ).pack(padx=16, pady=(0, 14))

    def _build_form_area(self, parent: ctk.CTkFrame) -> None:
        info = f"Dataset: {self._best.stem}  •  Hybrid ensemble  (dipilih otomatis)"
        ctk.CTkLabel(
            parent, text=info,
            anchor="w", text_color="gray55", font=ctk.CTkFont(size=11),
        ).grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="w")

        self._form_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self._form_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._form_frame.grid_columnconfigure(1, weight=1)
        self._rebuild_form(self._best.stem)

    def _rebuild_form(self, stem: str) -> None:
        for w in self._form_frame.winfo_children():
            w.destroy()
        self._entries.clear()
        self._field_readers.clear()
        self._field_resetters.clear()

        n = self._models.scaler.n_features_in_ if self._models and self._models.scaler else 5
        fields = load_field_config_from_yaml(stem) or _default_fields(n)
        slider_cfg = load_slider_config_from_yaml(stem)
        saved_inputs = load_last_inputs(stem)

        for idx, (label_text, key, hint) in enumerate(fields):
            r = idx * 2
            ctk.CTkLabel(
                self._form_frame, text=label_text,
                anchor="w", font=ctk.CTkFont(size=13),
            ).grid(row=r, column=0, padx=(20, 8), pady=(8, 0), sticky="w")

            if key in slider_cfg:
                self._build_slider_field(r, key, slider_cfg[key])
            else:
                entry = ctk.CTkEntry(self._form_frame, width=160, placeholder_text="0.0")
                entry.grid(row=r, column=1, padx=(0, 20), pady=(8, 0), sticky="ew")
                if key in saved_inputs:
                    entry.insert(0, str(saved_inputs[key]))
                self._entries[key] = entry
                self._field_readers[key] = lambda e=entry: float(e.get().strip())
                self._field_resetters[key] = lambda e=entry: e.delete(0, "end")

            ctk.CTkLabel(
                self._form_frame, text=hint,
                anchor="w", text_color="gray55", font=ctk.CTkFont(size=10),
                wraplength=380,
            ).grid(row=r + 1, column=0, columnspan=2, padx=(20, 20), pady=(0, 2), sticky="w")

    def _build_slider_field(self, row: int, key: str, cfg: dict) -> None:
        use_values = "values" in cfg
        if use_values:
            val_list: list[float] = [float(v) for v in cfg["values"]]
            default = float(cfg.get("default", val_list[0]))
            default_idx = min(range(len(val_list)), key=lambda i: abs(val_list[i] - default))
            current_idx = [default_idx]

            frame = ctk.CTkFrame(self._form_frame, fg_color="transparent")
            frame.grid(row=row, column=1, padx=(0, 20), pady=(8, 0), sticky="ew")
            frame.grid_columnconfigure(0, weight=1)

            val_lbl = ctk.CTkLabel(frame, text=f"{val_list[default_idx]:.2f}", width=46,
                                   font=ctk.CTkFont(size=12), text_color="#61afef")
            val_lbl.grid(row=0, column=1, padx=(6, 0))

            def make_discrete_cmd(curr, vals, lbl):
                def cmd(v):
                    n = len(vals) - 1
                    idx = min(max(round(v * n), 0), n)
                    curr[0] = idx
                    lbl.configure(text=f"{vals[idx]:.2f}")
                return cmd

            slider = ctk.CTkSlider(
                frame, from_=0, to=1,
                number_of_steps=len(val_list) - 1,
                command=make_discrete_cmd(current_idx, val_list, val_lbl),
            )
            slider.set(default_idx / max(len(val_list) - 1, 1))
            slider.grid(row=0, column=0, sticky="ew", pady=2)

            self._field_readers[key] = lambda curr=current_idx, vals=val_list: float(vals[curr[0]])
            self._field_resetters[key] = lambda s=slider, lbl=val_lbl, idx=default_idx, vals=val_list, n=max(len(val_list)-1, 1), curr=current_idx: (
                s.set(idx / n), lbl.configure(text=f"{vals[idx]:.2f}"), curr.__setitem__(0, idx)
            )
        else:
            lo = float(cfg.get("min", 0))
            hi = float(cfg.get("max", 100))
            step = float(cfg.get("step", 1))
            default = float(cfg.get("default", lo))
            n_steps = max(1, round((hi - lo) / step))
            current_val = [default]

            frame = ctk.CTkFrame(self._form_frame, fg_color="transparent")
            frame.grid(row=row, column=1, padx=(0, 20), pady=(8, 0), sticky="ew")
            frame.grid_columnconfigure(0, weight=1)

            fmt = ".0f" if step >= 1.0 else ".2f"
            val_lbl = ctk.CTkLabel(frame, text=f"{default:{fmt}}", width=46,
                                   font=ctk.CTkFont(size=12), text_color="#61afef")
            val_lbl.grid(row=0, column=1, padx=(6, 0))

            def make_cont_cmd(lo_, step_, hi_, fmt_, lbl, curr):
                def cmd(v):
                    snapped = lo_ + round((v - lo_) / step_) * step_
                    snapped = min(max(snapped, lo_), hi_)
                    curr[0] = snapped
                    lbl.configure(text=f"{snapped:{fmt_}}")
                return cmd

            slider = ctk.CTkSlider(
                frame, from_=lo, to=hi, number_of_steps=n_steps,
                command=make_cont_cmd(lo, step, hi, fmt, val_lbl, current_val),
            )
            slider.set(default)
            slider.grid(row=0, column=0, sticky="ew", pady=2)

            self._field_readers[key] = lambda curr=current_val: float(curr[0])
            self._field_resetters[key] = lambda s=slider, lbl=val_lbl, d=default, fmt_=fmt, curr=current_val: (
                s.set(d), lbl.configure(text=f"{d:{fmt_}}"), curr.__setitem__(0, d)
            )

    def _build_results_panel(self, parent) -> None:
        ctk.CTkLabel(
            parent, text="Hasil Estimasi & Validasi",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, pady=(14, 6), padx=20)

        self._error_label = ctk.CTkLabel(
            parent, text="", text_color="#e06c75",
            font=ctk.CTkFont(size=11), wraplength=440,
        )
        self._error_label.grid(row=1, column=0, pady=(0, 2), padx=20)

        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.grid(row=2, column=0, pady=(0, 10))
        est_state = "normal" if self._best else "disabled"
        ctk.CTkButton(
            btn_frame, text="Estimasi", width=140,
            command=self._on_estimate, state=est_state,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            btn_frame, text="Reset Form", width=140,
            fg_color="gray30", hover_color="gray40",
            command=self._on_reset,
        ).pack(side="left", padx=6)
        self._export_btn = ctk.CTkButton(
            btn_frame, text="Export", width=120,
            fg_color="gray30", hover_color="gray40",
            command=self._on_export, state="disabled",
        )
        self._export_btn.pack(side="left", padx=6)

        # ── Result card ──
        result_card = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        result_card.grid(row=3, column=0, padx=20, pady=(0, 8), sticky="ew")
        result_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            result_card, text="Estimasi Effort",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, pady=(14, 2))

        self._result_label = ctk.CTkLabel(
            result_card, text="—",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="#61afef",
        )
        self._result_label.grid(row=1, column=0, pady=(0, 0))

        self._unit_label = ctk.CTkLabel(
            result_card, text="person-months",
            text_color="gray55", font=ctk.CTkFont(size=11),
        )
        self._unit_label.grid(row=2, column=0, pady=(0, 4))

        self._confidence_label = ctk.CTkLabel(
            result_card, text="—",
            font=ctk.CTkFont(size=12), text_color="gray50",
        )
        self._confidence_label.grid(row=3, column=0, pady=(0, 6))

        ql_text, ql_color = _quality_info(self._best.mmre if self._best else None)
        self._quality_label = ctk.CTkLabel(
            result_card, text=ql_text,
            font=ctk.CTkFont(size=13, weight="bold"), text_color=ql_color,
        )
        self._quality_label.grid(row=4, column=0, pady=(0, 14))

        # ── Metric verification card ──
        metric_card = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        metric_card.grid(row=4, column=0, padx=20, pady=(0, 8), sticky="ew")
        metric_card.grid_columnconfigure(0, weight=0, minsize=90)
        metric_card.grid_columnconfigure(1, weight=0, minsize=80)
        metric_card.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            metric_card, text="Verifikasi Akurasi Model",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(12, 8), padx=16)

        b = self._best
        mae_text = f"{b.cv_mae_pm:.1f} pm" if b else "N/A"
        mmre_text = f"{b.mmre * 100:.1f}%" if (b and b.mmre is not None) else "N/A"
        pred25_text = f"{b.pred25 * 100:.1f}%" if (b and b.pred25 is not None) else "N/A"

        rows = [
            ("MAE", mae_text, "Rata-rata selisih prediksi dari nilai aktual"),
            ("MMRE", mmre_text, "Rata-rata error relatif  (baik: <25%)"),
            ("PRED(25)", pred25_text, "Prediksi dalam ±25% nilai aktual  (baik: >75%)"),
        ]
        for i, (metric, val, desc) in enumerate(rows):
            r = i + 1
            val_color = "#61afef" if val != "N/A" else "gray50"
            ctk.CTkLabel(
                metric_card, text=metric,
                font=ctk.CTkFont(size=12, weight="bold"), anchor="w",
            ).grid(row=r, column=0, padx=(16, 4), pady=4, sticky="w")
            ctk.CTkLabel(
                metric_card, text=val,
                font=ctk.CTkFont(size=12), text_color=val_color, anchor="w",
            ).grid(row=r, column=1, padx=(0, 8), pady=4, sticky="w")
            ctk.CTkLabel(
                metric_card, text=desc,
                font=ctk.CTkFont(size=10), text_color="gray55", anchor="w",
            ).grid(row=r, column=2, padx=(0, 16), pady=4, sticky="w")

        # bottom padding
        ctk.CTkLabel(metric_card, text="").grid(row=5, column=0, pady=(2, 6))

        # ── Calculator card ──
        calc = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        calc.grid(row=5, column=0, padx=20, pady=(0, 8), sticky="ew")
        calc.grid_columnconfigure(1, weight=1)
        self._build_calculator(calc)

        # ── Detail Teknis (collapsible) ──
        detail_section = ctk.CTkFrame(parent, fg_color="transparent")
        detail_section.grid(row=6, column=0, padx=20, pady=(0, 8), sticky="ew")
        detail_section.grid_columnconfigure(0, weight=1)
        self._build_detail_section(detail_section)

        # ── Model comparison (shown after first estimation) ──
        self._comparison_section = ctk.CTkFrame(parent, fg_color="transparent")
        self._comparison_section.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="ew")
        self._comparison_section.grid_columnconfigure(0, weight=1)
        self._build_comparison_section(self._comparison_section)
        self._comparison_section.grid_remove()

    def _build_comparison_section(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent, text="Perbandingan Model",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, pady=(12, 6), sticky="w")

        # Checkbox row
        cb_frame = ctk.CTkFrame(parent, fg_color="transparent")
        cb_frame.grid(row=1, column=0, pady=(0, 8), sticky="ew")
        self._model_check_vars.clear()
        for i, name in enumerate(_COMPARISON_MODELS):
            var = ctk.BooleanVar(value=False)
            self._model_check_vars[name] = var
            cb = ctk.CTkCheckBox(
                cb_frame, text=name, variable=var, font=ctk.CTkFont(size=11),
                command=self._on_comparison_change,
            )
            cb.grid(row=0, column=i, padx=(0, 12), sticky="w")

        # Table frame (rebuilt on each refresh)
        self._comparison_table_frame = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        self._comparison_table_frame.grid(row=2, column=0, pady=(0, 8), sticky="ew")

        # Chart frame (rebuilt on each refresh)
        self._comparison_chart_frame = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        self._comparison_chart_frame.grid(row=3, column=0, pady=(0, 0), sticky="ew")
        self._comparison_chart_canvas: FigureCanvasTkAgg | None = None

    def _on_comparison_change(self) -> None:
        selected = [n for n, v in self._model_check_vars.items() if v.get()]
        # Prevent deselecting the last model
        if not selected:
            for name, var in self._model_check_vars.items():
                if self._last_result and name in self._last_result.get("predictions", {}):
                    var.set(True)
                    selected = [name]
                    break
        if self._last_result:
            self._refresh_comparison(selected)

    def _refresh_comparison(self, selected: list[str]) -> None:
        predictions = self._last_result["predictions"] if self._last_result else {}
        self._render_comparison_table(self._comparison_table_frame, selected, predictions)
        self._render_comparison_chart(self._comparison_chart_frame, selected, predictions)

    def _render_comparison_table(
        self, parent: ctk.CTkFrame, selected: list[str], predictions: dict
    ) -> None:
        for w in parent.winfo_children():
            w.destroy()

        metrics = self._metrics_cache
        col_models = [m for m in selected if m in predictions]
        if not col_models:
            return

        header_font = ctk.CTkFont(size=11, weight="bold")
        cell_font = ctk.CTkFont(size=11)
        gray = "gray55"

        # Header row
        ctk.CTkLabel(parent, text="", font=header_font).grid(row=0, column=0, padx=(12, 8), pady=(10, 4), sticky="w")
        for c, name in enumerate(col_models):
            ctk.CTkLabel(parent, text=name[:12], font=header_font).grid(
                row=0, column=c + 1, padx=8, pady=(10, 4)
            )

        row_defs = [
            ("Prediksi (pm)", lambda m: f"{predictions[m]:.2f}" if m in predictions else "—"),
            ("MAE",  lambda m: f"{metrics.get(_METRIC_KEYS[m]['mae']):.1f}" if m in _METRIC_KEYS and _METRIC_KEYS[m]['mae'] and _METRIC_KEYS[m]['mae'] in metrics else "—"),
            ("RMSE", lambda m: f"{metrics.get(_METRIC_KEYS[m]['rmse']):.1f}" if m in _METRIC_KEYS and _METRIC_KEYS[m]['rmse'] and _METRIC_KEYS[m]['rmse'] in metrics else "—"),
            ("R²",   lambda m: f"{metrics.get(_METRIC_KEYS[m]['r2']):.3f}" if m in _METRIC_KEYS and _METRIC_KEYS[m]['r2'] and _METRIC_KEYS[m]['r2'] in metrics else "—"),
        ]
        for r, (label, getter) in enumerate(row_defs):
            ctk.CTkLabel(parent, text=label, font=cell_font, text_color=gray, anchor="w").grid(
                row=r + 1, column=0, padx=(12, 8), pady=3, sticky="w"
            )
            for c, model in enumerate(col_models):
                ctk.CTkLabel(parent, text=getter(model), font=cell_font, anchor="e").grid(
                    row=r + 1, column=c + 1, padx=8, pady=3, sticky="e"
                )
        ctk.CTkLabel(parent, text="").grid(row=len(row_defs) + 1, column=0, pady=(0, 6))

    def _render_comparison_chart(
        self, parent: ctk.CTkFrame, selected: list[str], predictions: dict
    ) -> None:
        for w in parent.winfo_children():
            w.destroy()
        if self._comparison_chart_canvas:
            plt.close("all")
            self._comparison_chart_canvas = None

        col_models = [m for m in selected if m in predictions]
        if not col_models:
            return

        vals = [predictions[m] for m in col_models]
        labels = [m[:10] for m in col_models]

        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        fig.patch.set_facecolor("#2b2b2b")
        ax.set_facecolor("#2b2b2b")
        bars = ax.bar(labels, vals, color="#61afef", width=0.5)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color="white")
        ax.set_ylabel("person-months", fontsize=8, color="#888")
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        fig.tight_layout(pad=0.6)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._comparison_chart_canvas = canvas

    def _build_calculator(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent, text="Kalkulator Biaya & Tim",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(12, 6), padx=20)

        ctk.CTkLabel(
            parent, text="Gaji Bulanan (Rp)", anchor="w", font=ctk.CTkFont(size=11),
        ).grid(row=1, column=0, padx=(16, 6), pady=4, sticky="w")
        self._gaji_entry = ctk.CTkEntry(parent, placeholder_text="12000000", width=160)
        self._gaji_entry.grid(row=1, column=1, padx=(0, 16), pady=4, sticky="ew")
        self._gaji_entry.bind("<KeyRelease>", lambda _: self._update_calculator())

        ctk.CTkLabel(
            parent, text="Target Durasi (bulan)", anchor="w", font=ctk.CTkFont(size=11),
        ).grid(row=2, column=0, padx=(16, 6), pady=4, sticky="w")
        self._durasi_entry = ctk.CTkEntry(parent, placeholder_text="6", width=160)
        self._durasi_entry.grid(row=2, column=1, padx=(0, 16), pady=4, sticky="ew")
        self._durasi_entry.bind("<KeyRelease>", lambda _: self._update_calculator())

        ctk.CTkLabel(
            parent, text="Total Biaya", anchor="w", font=ctk.CTkFont(size=11),
        ).grid(row=3, column=0, padx=(16, 6), pady=4, sticky="w")
        self._biaya_label = ctk.CTkLabel(
            parent, text="—", anchor="e", font=ctk.CTkFont(size=11), text_color="#98c379",
        )
        self._biaya_label.grid(row=3, column=1, padx=(0, 16), pady=4, sticky="e")

        ctk.CTkLabel(
            parent, text="Kebutuhan Tim", anchor="w", font=ctk.CTkFont(size=11),
        ).grid(row=4, column=0, padx=(16, 6), pady=(4, 14), sticky="w")
        self._tim_label = ctk.CTkLabel(
            parent, text="—", anchor="e", font=ctk.CTkFont(size=11), text_color="#98c379",
        )
        self._tim_label.grid(row=4, column=1, padx=(0, 16), pady=(4, 14), sticky="e")

    def _build_detail_section(self, parent: ctk.CTkFrame) -> None:
        self._detail_toggle_btn = ctk.CTkButton(
            parent, text="▶  Detail Teknis",
            fg_color="gray20", hover_color="gray30",
            anchor="w", font=ctk.CTkFont(size=12),
            command=self._toggle_detail,
        )
        self._detail_toggle_btn.grid(row=0, column=0, sticky="ew", pady=(0, 0))

        self._detail_content = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        self._detail_content.grid_columnconfigure(1, weight=1)
        # hidden by default — grid_remove after first layout
        self._detail_content.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._populate_detail_content()
        self._detail_content.grid_remove()

    def _populate_detail_content(self) -> None:
        for w in self._detail_content.winfo_children():
            w.destroy()

        b = self._best
        if b is None:
            return

        hw = self._models.hybrid_weights if self._models else None
        model_display = (
            f"Hybrid (LR + XGB + LSTM)"
            if hw else b.model_name
        )
        rows = [
            ("Dataset", b.stem),
            ("Model", model_display),
            ("Dilatih pada", f"{b.trained_on_rows} proyek"),
            ("Satuan effort", b.effort_unit),
            ("CV MAE (asli)", f"{b.cv_mae:.2f} {b.effort_unit.split('-')[-1]}"),
            ("CV MAE (norm.)", f"{b.cv_mae_pm:.2f} person-months"),
        ]
        if hw:
            rows.append(("  bobot LR", f"{hw.get('lr', 0):.3f}"))
            rows.append(("  bobot XGB", f"{hw.get('xgb', 0):.3f}"))
            rows.append(("  bobot LSTM", f"{hw.get('lstm', 0):.3f}"))
        elif b.hyperparams:
            for k, v in b.hyperparams.items():
                rows.append((f"  {k}", str(v)))

        for i, (label, val) in enumerate(rows):
            ctk.CTkLabel(
                self._detail_content, text=label,
                font=ctk.CTkFont(size=11), anchor="w", text_color="gray60",
            ).grid(row=i, column=0, padx=(14, 6), pady=2, sticky="w")
            ctk.CTkLabel(
                self._detail_content, text=val,
                font=ctk.CTkFont(size=11), anchor="w",
            ).grid(row=i, column=1, padx=(0, 14), pady=2, sticky="w")

        ctk.CTkLabel(self._detail_content, text="").grid(
            row=len(rows), column=0, pady=(2, 6)
        )

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _toggle_detail(self) -> None:
        self._detail_visible = not self._detail_visible
        if self._detail_visible:
            self._detail_content.grid()
            self._detail_toggle_btn.configure(text="▼  Detail Teknis")
        else:
            self._detail_content.grid_remove()
            self._detail_toggle_btn.configure(text="▶  Detail Teknis")

    def _on_estimate(self) -> None:
        self._error_label.configure(text="")
        if self._best is None or self._models is None:
            return

        stem = self._best.stem
        n = self._models.scaler.n_features_in_ if self._models.scaler else 5
        fields = load_field_config_from_yaml(stem) or _default_fields(n)

        values = []
        for label_text, key, _ in fields:
            reader = self._field_readers.get(key)
            if reader is None:
                self._error_label.configure(text=f"Field '{label_text}' tidak ditemukan.")
                return
            try:
                values.append(reader())
            except (ValueError, KeyError):
                self._error_label.configure(
                    text=f"Input tidak valid pada '{label_text}'. Masukkan angka."
                )
                return

        X = np.array(values).reshape(1, -1)
        X_scaled = self._models.scaler.transform(X)

        pred = run_hybrid_estimate(X_scaled, self._models, self._best.effort_unit)
        if pred is None:
            pred = run_estimate(
                self._best.model_name, X_scaled, self._models, self._best.effort_unit
            )
        if pred is None:
            self._error_label.configure(text="Model tidak dapat menghasilkan prediksi.")
            return

        self._current_estimate = pred
        self._result_label.configure(text=f"{pred:.2f}")

        # Collect all individual model predictions for export
        all_preds: dict[str, float] = {}
        for model_name in ["SVR", "Linear Regression", "Random Forest", "XGBoost", "LSTM"]:
            p = run_estimate(model_name, X_scaled, self._models, self._best.effort_unit)
            if p is not None:
                all_preds[model_name] = round(p, 4)
        hybrid_p = run_hybrid_estimate(X_scaled, self._models, self._best.effort_unit)
        if hybrid_p is not None:
            all_preds["Hybrid"] = round(hybrid_p, 4)
        self._last_result = {
            "inputs": {key: reader() for key, reader in self._field_readers.items()},
            "predictions": all_preds,
        }
        self._export_btn.configure(state="normal")
        save_last_inputs(self._best.stem, self._last_result["inputs"])

        # Load metrics for comparison table
        metrics_path = os.path.join(MODELS_DIR, f"{self._best.stem}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as _f:
                self._metrics_cache = json.load(_f)

        # Show comparison section with all available models checked
        for name, var in self._model_check_vars.items():
            var.set(name in all_preds)
        selected = [n for n, v in self._model_check_vars.items() if v.get()]
        self._refresh_comparison(selected)
        self._comparison_section.grid()

        unit_hint = ""
        if self._best.effort_unit == "person-hours":
            unit_hint = f"  (dari person-hours ÷ {HOURS_PER_MONTH})"
        self._unit_label.configure(text=f"person-months{unit_hint}")

        # confidence range from CV MAE (normalized)
        conf = self._best.cv_mae_pm
        self._confidence_label.configure(text=f"± {conf:.1f} person-months")

        self._update_calculator()

    def _on_reset(self) -> None:
        for resetter in self._field_resetters.values():
            resetter()
        self._current_estimate = None
        self._last_result = None
        self._metrics_cache = {}
        if hasattr(self, "_export_btn"):
            self._export_btn.configure(state="disabled")
        if hasattr(self, "_comparison_section"):
            self._comparison_section.grid_remove()
        if hasattr(self, "_error_label"):
            self._error_label.configure(text="")
        if hasattr(self, "_result_label"):
            self._result_label.configure(text="—")
        if hasattr(self, "_unit_label"):
            self._unit_label.configure(text="person-months")
        if hasattr(self, "_confidence_label"):
            self._confidence_label.configure(text="—")
        if hasattr(self, "_gaji_entry"):
            self._gaji_entry.delete(0, "end")
        if hasattr(self, "_durasi_entry"):
            self._durasi_entry.delete(0, "end")
        if hasattr(self, "_biaya_label"):
            self._biaya_label.configure(text="—")
        if hasattr(self, "_tim_label"):
            self._tim_label.configure(text="—")

    def _on_export(self) -> None:
        if self._last_result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")],
            title="Simpan Hasil Estimasi",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                self._export_json(path, self._last_result["inputs"], self._last_result["predictions"])
            else:
                self._export_csv(path, self._last_result["inputs"], self._last_result["predictions"])
        except OSError as exc:
            self._error_label.configure(text=f"Gagal menyimpan file: {exc}")

    def _export_csv(self, path: str, inputs: dict, predictions: dict) -> None:
        timestamp = datetime.datetime.now().isoformat()
        row = {
            **{k: v for k, v in inputs.items()},
            **{f"pred_{name.replace(' ', '_').lower()}": v for name, v in predictions.items()},
            "timestamp": timestamp,
        }
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    def _export_json(self, path: str, inputs: dict, predictions: dict) -> None:
        data = {
            "inputs": inputs,
            "predictions": predictions,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _update_calculator(self) -> None:
        if self._current_estimate is None:
            return
        try:
            gaji = float(self._gaji_entry.get().strip())
            durasi = float(self._durasi_entry.get().strip())
        except ValueError:
            return
        if durasi <= 0:
            return
        total_biaya = self._current_estimate * gaji
        kebutuhan_tim = math.ceil(self._current_estimate / durasi)
        self._biaya_label.configure(text=f"Rp {total_biaya:,.0f}")
        self._tim_label.configure(text=f"{kebutuhan_tim} orang")

    def _on_close(self) -> None:
        self.quit()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
