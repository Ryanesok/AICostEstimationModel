import glob
import math
import os

import customtkinter as ctk
import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODELS_DIR = "models"
CHART_BG = "#1c1c1c"
CHART_FG = "#ececec"
COLOR_SVR = "#61afef"
COLOR_LR = "#98c379"

CONFIG_FILE = "dataset_config.yaml"
HOURS_PER_MONTH = 160


def _default_fields(n: int) -> list[tuple]:
    return [(f"Feature {i + 1}", f"f{i}", f"Input fitur ke-{i + 1}") for i in range(n)]


def load_field_config_from_yaml(stem: str) -> list[tuple] | None:
    """Load feature columns with labels and hints from dataset_config.yaml."""
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


def get_effort_unit(stem: str) -> str:
    """Return effort_unit for a dataset stem ('person-months' or 'person-hours')."""
    if not os.path.exists(CONFIG_FILE):
        return "person-months"
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get("datasets", {}).get(stem.lower(), {})
    return cfg.get("effort_unit", "person-months")


def to_person_months(value: float, unit: str) -> float:
    """Convert raw model output to person-months."""
    if unit == "person-hours":
        return value / HOURS_PER_MONTH
    return value


def load_slider_config_from_yaml(stem: str) -> dict:
    """Return {col: slider_config} for fields that should use a slider."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get("datasets", {}).get(stem.lower(), {})
    return cfg.get("sliders", {})


def discover_models() -> list[str]:
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "svr_*.pkl")))
    return [os.path.basename(p)[4:-4] for p in paths]  # strip "svr_" and ".pkl"


def load_model_trio(stem: str):
    try:
        svr = joblib.load(os.path.join(MODELS_DIR, f"svr_{stem}.pkl"))
        lr = joblib.load(os.path.join(MODELS_DIR, f"linreg_{stem}.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, f"scaler_{stem}.pkl"))
        return svr, lr, scaler, None
    except FileNotFoundError as exc:
        return None, None, None, str(exc)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OpenSpec - AI Cost Estimator")
        self.geometry("1000x780")
        self.resizable(False, False)

        self._stems = discover_models()
        self._svr = None
        self._lr = None
        self._scaler = None
        self._pred_svr: float | None = None
        self._pred_lr: float | None = None
        self._effort_unit: str = "person-months"
        self._entries: dict[str, ctk.CTkEntry] = {}
        self._field_readers: dict[str, callable] = {}
        self._field_resetters: dict[str, callable] = {}
        self._canvas = None
        self._fig = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Left panel (scrollable) ───────────────────────────────────────────
        left = ctk.CTkScrollableFrame(self, corner_radius=12, width=460)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        left.grid_columnconfigure(1, weight=1)
        self._left = left

        ctk.CTkLabel(
            left,
            text="Estimasi Biaya Perangkat Lunak",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(14, 8), padx=20)

        if not self._stems:
            self._build_error_banner(left)
        else:
            self._build_form_area(left)

        # ── Right panel: chart ────────────────────────────────────────────────
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            right,
            text="Visualisasi Perbandingan Model",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, pady=(18, 6))
        self._chart_frame = right
        self._draw_empty_chart()

    def _build_form_area(self, parent: ctk.CTkFrame) -> None:
        # Dataset selector
        ctk.CTkLabel(
            parent, text="Pilih Dataset", anchor="w", font=ctk.CTkFont(size=13)
        ).grid(row=1, column=0, padx=(20, 8), pady=(6, 4), sticky="w")
        self._dataset_var = ctk.StringVar(value=self._stems[0])
        ctk.CTkOptionMenu(
            parent,
            values=self._stems,
            variable=self._dataset_var,
            command=self._on_dataset_change,
        ).grid(row=1, column=1, padx=(0, 20), pady=(6, 4), sticky="ew")

        # Dynamic form container
        self._form_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self._form_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._form_frame.grid_columnconfigure(1, weight=1)

        # Model toggle
        ctk.CTkLabel(
            parent, text="Model Prediksi", anchor="w", font=ctk.CTkFont(size=13)
        ).grid(row=3, column=0, padx=(20, 8), pady=(8, 4), sticky="w")
        self._model_var = ctk.StringVar(value="SVR")
        ctk.CTkSegmentedButton(
            parent,
            values=["SVR", "Linear Regression"],
            variable=self._model_var,
            command=self._on_model_toggle,
        ).grid(row=3, column=1, padx=(0, 20), pady=(8, 4), sticky="ew")

        # Inline error
        self._error_label = ctk.CTkLabel(
            parent, text="", text_color="#e06c75", font=ctk.CTkFont(size=11)
        )
        self._error_label.grid(row=4, column=0, columnspan=2, pady=(4, 0), padx=20)

        # Buttons
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(8, 4))
        ctk.CTkButton(btn_frame, text="Estimasi", width=130, command=self._on_estimate).pack(
            side="left", padx=6
        )
        ctk.CTkButton(
            btn_frame,
            text="Reset Form",
            width=130,
            fg_color="gray30",
            hover_color="gray40",
            command=self._on_reset,
        ).pack(side="left", padx=6)

        # Result area
        result_frame = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        result_frame.grid(row=6, column=0, columnspan=2, padx=20, pady=(4, 4), sticky="ew")
        result_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            result_frame, text="Hasil Estimasi", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, pady=(10, 2))
        self._result_label = ctk.CTkLabel(
            result_frame,
            text="—",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#61afef",
        )
        self._result_label.grid(row=1, column=0, pady=(2, 2))
        self._unit_label = ctk.CTkLabel(
            result_frame, text="", font=ctk.CTkFont(size=11), text_color="gray60"
        )
        self._unit_label.grid(row=2, column=0, pady=(0, 8))

        # ── Calculator ────────────────────────────────────────────────────────
        calc = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        calc.grid(row=7, column=0, columnspan=2, padx=20, pady=(0, 12), sticky="ew")
        calc.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            calc, text="Kalkulator Biaya & Tim", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(10, 6), padx=20)

        ctk.CTkLabel(
            calc, text="Gaji Bulanan (Rp)", anchor="w", font=ctk.CTkFont(size=11)
        ).grid(row=1, column=0, padx=(16, 6), pady=4, sticky="w")
        self._gaji_entry = ctk.CTkEntry(calc, placeholder_text="12000000", width=140)
        self._gaji_entry.grid(row=1, column=1, padx=(0, 16), pady=4, sticky="ew")
        self._gaji_entry.bind("<KeyRelease>", lambda _: self._update_calculator())

        ctk.CTkLabel(
            calc, text="Target Durasi (bulan)", anchor="w", font=ctk.CTkFont(size=11)
        ).grid(row=2, column=0, padx=(16, 6), pady=4, sticky="w")
        self._durasi_entry = ctk.CTkEntry(calc, placeholder_text="6", width=140)
        self._durasi_entry.grid(row=2, column=1, padx=(0, 16), pady=4, sticky="ew")
        self._durasi_entry.bind("<KeyRelease>", lambda _: self._update_calculator())

        ctk.CTkLabel(
            calc, text="Total Biaya", anchor="w", font=ctk.CTkFont(size=11)
        ).grid(row=3, column=0, padx=(16, 6), pady=4, sticky="w")
        self._biaya_label = ctk.CTkLabel(
            calc, text="—", anchor="e", font=ctk.CTkFont(size=11), text_color="#98c379"
        )
        self._biaya_label.grid(row=3, column=1, padx=(0, 16), pady=4, sticky="e")

        ctk.CTkLabel(
            calc, text="Kebutuhan Tim", anchor="w", font=ctk.CTkFont(size=11)
        ).grid(row=4, column=0, padx=(16, 6), pady=(4, 12), sticky="w")
        self._tim_label = ctk.CTkLabel(
            calc, text="—", anchor="e", font=ctk.CTkFont(size=11), text_color="#98c379"
        )
        self._tim_label.grid(row=4, column=1, padx=(0, 16), pady=(4, 12), sticky="e")

        # Load initial model and build form
        self._effort_unit = get_effort_unit(self._stems[0])
        self._load_dataset(self._stems[0])
        self._rebuild_form(self._stems[0])

    def _build_error_banner(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(
            frame,
            text="Model Tidak Tersedia",
            text_color="#e06c75",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=(14, 6))
        ctk.CTkLabel(
            frame,
            text=(
                "Tidak ada model terlatih di folder 'models/'.\n\n"
                "Jalankan langkah berikut:\n"
                "  1. python downloader.py\n"
                "  2. python model_pipeline.py"
            ),
            text_color="gray70",
            font=ctk.CTkFont(size=12),
            justify="left",
        ).pack(padx=16, pady=(0, 14))

    def _rebuild_form(self, stem: str) -> None:
        for widget in self._form_frame.winfo_children():
            widget.destroy()
        self._entries.clear()
        self._field_readers.clear()
        self._field_resetters.clear()

        n_features = self._scaler.n_features_in_ if self._scaler else 5
        fields = load_field_config_from_yaml(stem) or _default_fields(n_features)
        slider_cfg = load_slider_config_from_yaml(stem)

        for idx, (label_text, key, hint) in enumerate(fields):
            r = idx * 2
            ctk.CTkLabel(
                self._form_frame, text=label_text, anchor="w", font=ctk.CTkFont(size=13)
            ).grid(row=r, column=0, padx=(20, 8), pady=(8, 0), sticky="w")

            if key in slider_cfg:
                self._build_slider_field(r, key, slider_cfg[key])
            else:
                entry = ctk.CTkEntry(self._form_frame, width=160, placeholder_text="0.0")
                entry.grid(row=r, column=1, padx=(0, 20), pady=(8, 0), sticky="ew")
                self._entries[key] = entry
                self._field_readers[key] = lambda e=entry: float(e.get().strip())
                self._field_resetters[key] = lambda e=entry: e.delete(0, "end")

            ctk.CTkLabel(
                self._form_frame,
                text=hint,
                anchor="w",
                text_color="gray55",
                font=ctk.CTkFont(size=10),
                wraplength=380,
            ).grid(row=r + 1, column=0, columnspan=2, padx=(20, 20), pady=(0, 2), sticky="w")

    def _load_dataset(self, stem: str) -> None:
        self._svr, self._lr, self._scaler, err = load_model_trio(stem)
        if err and hasattr(self, "_error_label"):
            self._error_label.configure(text=f"Gagal memuat model '{stem}': {err}")

    def _build_slider_field(self, row: int, key: str, cfg: dict) -> None:
        """Create a CTkSlider widget in the form, wiring _field_readers and _field_resetters."""
        use_values = "values" in cfg  # discrete snap mode

        if use_values:
            val_list: list[float] = [float(v) for v in cfg["values"]]
            default = float(cfg.get("default", val_list[0]))
            default_idx = min(range(len(val_list)), key=lambda i: abs(val_list[i] - default))
            current_idx = [default_idx]  # mutable ref

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
            self._field_resetters[key] = lambda s=slider, lbl=val_lbl, idx=default_idx, vals=val_list, n=max(len(val_list)-1,1), curr=current_idx: (
                s.set(idx / n), lbl.configure(text=f"{vals[idx]:.2f}"), curr.__setitem__(0, idx)
            )
        else:
            lo = float(cfg.get("min", 0))
            hi = float(cfg.get("max", 100))
            step = float(cfg.get("step", 1))
            default = float(cfg.get("default", lo))
            n_steps = max(1, round((hi - lo) / step))
            current_val = [default]  # mutable ref

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

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _unit_label_text(self, model_name: str) -> str:
        if self._effort_unit == "person-hours":
            return f"Person-Months  (dari person-hours ÷ {HOURS_PER_MONTH})  [{model_name}]"
        return f"Person-Months  [{model_name}]"

    def _on_dataset_change(self, choice: str) -> None:
        self._effort_unit = get_effort_unit(choice)
        self._load_dataset(choice)
        self._rebuild_form(choice)
        self._on_reset()

    def _on_model_toggle(self, _value: str) -> None:
        if self._pred_svr is None:
            return
        pred = self._pred_svr if self._model_var.get() == "SVR" else self._pred_lr
        self._result_label.configure(text=f"{pred:.2f}")
        self._unit_label.configure(text=self._unit_label_text(self._model_var.get()))
        self._update_calculator()

    def _on_estimate(self) -> None:
        self._error_label.configure(text="")
        stem = self._dataset_var.get()
        n_features = self._scaler.n_features_in_ if self._scaler else 5
        fields = load_field_config_from_yaml(stem) or _default_fields(n_features)

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
                    text=f"Input tidak valid pada kolom '{label_text}'. Masukkan angka."
                )
                return

        X = np.array(values).reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        raw_svr = float(self._svr.predict(X_scaled)[0])
        raw_lr = float(self._lr.predict(X_scaled)[0])

        unit = get_effort_unit(stem)
        self._effort_unit = unit
        self._pred_svr = to_person_months(raw_svr, unit)
        self._pred_lr = to_person_months(raw_lr, unit)

        selected = self._model_var.get()
        prediction = self._pred_svr if selected == "SVR" else self._pred_lr
        self._result_label.configure(text=f"{prediction:.2f}")
        self._unit_label.configure(text=self._unit_label_text(selected))
        self._update_chart(self._pred_svr, self._pred_lr)
        self._update_calculator()

    def _on_reset(self) -> None:
        for resetter in self._field_resetters.values():
            resetter()
        self._pred_svr = None
        self._pred_lr = None
        if hasattr(self, "_error_label"):
            self._error_label.configure(text="")
        if hasattr(self, "_result_label"):
            self._result_label.configure(text="—")
        if hasattr(self, "_unit_label"):
            self._unit_label.configure(text="")
        if hasattr(self, "_gaji_entry"):
            self._gaji_entry.delete(0, "end")
        if hasattr(self, "_durasi_entry"):
            self._durasi_entry.delete(0, "end")
        if hasattr(self, "_biaya_label"):
            self._biaya_label.configure(text="—")
        if hasattr(self, "_tim_label"):
            self._tim_label.configure(text="—")
        self._draw_empty_chart()

    def _on_close(self) -> None:
        plt.close("all")
        self.quit()
        self.destroy()

    def _update_calculator(self) -> None:
        if self._pred_svr is None:
            return
        pred = self._pred_svr if self._model_var.get() == "SVR" else self._pred_lr
        try:
            gaji = float(self._gaji_entry.get().strip())
            durasi = float(self._durasi_entry.get().strip())
        except ValueError:
            return
        if durasi <= 0:
            return
        total_biaya = pred * gaji
        kebutuhan_tim = math.ceil(pred / durasi)
        self._biaya_label.configure(text=f"Rp {total_biaya:,.0f}")
        self._tim_label.configure(text=f"{kebutuhan_tim} orang")

    # ── Chart ──────────────────────────────────────────────────────────────────

    def _draw_empty_chart(self) -> None:
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        if self._fig:
            plt.close(self._fig)
        fig, ax = plt.subplots(figsize=(4.2, 5.8), facecolor=CHART_BG)
        ax.set_facecolor(CHART_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.5, 0.5,
            "Tekan 'Estimasi'\nuntuk melihat chart",
            ha="center", va="center",
            color="gray", fontsize=11,
            transform=ax.transAxes,
        )
        self._embed_chart(fig)

    def _update_chart(self, pred_svr: float, pred_lr: float) -> None:
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        if self._fig:
            plt.close(self._fig)

        fig, ax = plt.subplots(figsize=(4.2, 5.8), facecolor=CHART_BG)
        ax.set_facecolor(CHART_BG)
        fig.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.10)

        values = [pred_svr, pred_lr]
        colors = [COLOR_SVR, COLOR_LR]
        bars = ax.barh(
            ["SVR", "Linear\nRegression"], values, color=colors, height=0.45, edgecolor="none"
        )
        for bar, val in zip(bars, values):
            ax.text(
                val + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center", ha="left",
                color=CHART_FG, fontsize=11, fontweight="bold",
            )

        xlabel = (
            f"Person-Months (÷{HOURS_PER_MONTH} dari person-hours)"
            if self._effort_unit == "person-hours"
            else "Person-Months"
        )
        ax.set_xlabel(xlabel, color="#b3b3b3", fontsize=9)
        ax.set_title("Prediksi Effort", color=CHART_FG, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="#b3b3b3", labelsize=10)
        ax.set_xlim(0, max(values) * 1.35 + 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.grid(True, color="#404040", linewidth=0.5)
        ax.set_axisbelow(True)
        self._embed_chart(fig)

    def _embed_chart(self, fig) -> None:
        canvas = FigureCanvasTkAgg(fig, master=self._chart_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=1, column=0, padx=10, pady=(0, 14), sticky="nsew")
        self._canvas = canvas
        self._fig = fig


if __name__ == "__main__":
    app = App()
    app.mainloop()
