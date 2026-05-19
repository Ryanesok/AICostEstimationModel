import glob
import json
import math
import os

import customtkinter as ctk
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODELS_DIR = "models"
CHART_BG = "#1c1c1c"
CHART_FG = "#ececec"
COLOR_HIGHLIGHT = "#4a9eff"
COLOR_MUTED = "#555555"

CONFIG_FILE = "dataset_config.yaml"
HOURS_PER_MONTH = 160

MODEL_NAMES = ["SVR", "Linear Regression", "Random Forest", "XGBoost", "LSTM", "Hybrid"]


# ── LSTM model definition (must match model_pipeline.py) ──────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_fields(n: int) -> list[tuple]:
    return [(f"Feature {i + 1}", f"f{i}", f"Input fitur ke-{i + 1}") for i in range(n)]


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


def get_effort_unit(stem: str) -> str:
    if not os.path.exists(CONFIG_FILE):
        return "person-months"
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get("datasets", {}).get(stem.lower(), {})
    return cfg.get("effort_unit", "person-months")


def to_person_months(value: float, unit: str) -> float:
    if unit == "person-hours":
        return value / HOURS_PER_MONTH
    return value


def load_slider_config_from_yaml(stem: str) -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get("datasets", {}).get(stem.lower(), {})
    return cfg.get("sliders", {})


def discover_models() -> list[str]:
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "svr_*.pkl")))
    return [os.path.basename(p)[4:-4] for p in paths]


def load_model_trio(stem: str):
    try:
        svr = joblib.load(os.path.join(MODELS_DIR, f"svr_{stem}.pkl"))
        lr = joblib.load(os.path.join(MODELS_DIR, f"linreg_{stem}.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, f"scaler_{stem}.pkl"))
        return svr, lr, scaler, None
    except FileNotFoundError as exc:
        return None, None, None, str(exc)


def _load_lstm(stem: str) -> LSTMModel | None:
    arch_path = os.path.join(MODELS_DIR, f"lstm_arch_{stem}.json")
    pt_path = os.path.join(MODELS_DIR, f"lstm_{stem}.pt")
    if not os.path.exists(arch_path) or not os.path.exists(pt_path):
        return None
    with open(arch_path, "r") as f:
        arch = json.load(f)
    model = LSTMModel(arch["input_size"])
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _load_hybrid_weights(stem: str) -> dict | None:
    metrics_path = os.path.join(MODELS_DIR, f"{stem}_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics.get("hybrid_weights")


# ── App ───────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OpenSpec - AI Cost Estimator")
        self.geometry("1400x780")
        self.resizable(False, False)

        self._stems = discover_models()
        self._svr = None
        self._lr = None
        self._scaler = None
        self._rf = None
        self._xgb = None
        self._lstm = None
        self._hybrid_weights: dict | None = None
        self._all_preds: dict[str, float] = {}
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
        self.grid_columnconfigure(0, weight=0, minsize=420)
        self.grid_columnconfigure(1, weight=0, minsize=380)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(self, corner_radius=12, width=400)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        left.grid_columnconfigure(1, weight=1)
        self._left = left

        ctk.CTkLabel(
            left,
            text="Input Fitur",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(14, 8), padx=20)

        if not self._stems:
            self._build_error_banner(left)
        else:
            self._build_form_area(left)

        middle = ctk.CTkFrame(self, corner_radius=12)
        middle.grid(row=0, column=1, padx=(8, 8), pady=16, sticky="nsew")
        self._middle = middle
        self._build_middle_panel(middle)

        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=2, padx=(8, 16), pady=16, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            right,
            text="Visualisasi Perbandingan Model",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, pady=(18, 6))
        self._chart_frame = right
        self._draw_empty_chart()

        if self._stems:
            self._effort_unit = get_effort_unit(self._stems[0])
            self._load_dataset(self._stems[0])
            self._rebuild_form(self._stems[0])

    def _build_form_area(self, parent: ctk.CTkFrame) -> None:
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

        self._form_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self._form_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._form_frame.grid_columnconfigure(1, weight=1)

    def _build_middle_panel(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            parent,
            text="Model & Hasil Estimasi",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(14, 8), padx=20)

        ctk.CTkLabel(
            parent, text="Model Prediksi", anchor="w", font=ctk.CTkFont(size=13)
        ).grid(row=1, column=0, padx=(20, 8), pady=(8, 4), sticky="w")
        self._model_var = ctk.StringVar(value="SVR")
        ctk.CTkOptionMenu(
            parent,
            values=MODEL_NAMES,
            variable=self._model_var,
            command=self._on_model_change,
        ).grid(row=1, column=1, padx=(0, 20), pady=(8, 4), sticky="ew")

        self._error_label = ctk.CTkLabel(
            parent, text="", text_color="#e06c75", font=ctk.CTkFont(size=11),
            wraplength=340,
        )
        self._error_label.grid(row=2, column=0, columnspan=2, pady=(4, 0), padx=20)

        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(8, 4))
        est_state = "normal" if self._stems else "disabled"
        ctk.CTkButton(
            btn_frame, text="Estimasi", width=130, command=self._on_estimate, state=est_state
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            btn_frame,
            text="Reset Form",
            width=130,
            fg_color="gray30",
            hover_color="gray40",
            command=self._on_reset,
        ).pack(side="left", padx=6)

        result_frame = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        result_frame.grid(row=4, column=0, columnspan=2, padx=16, pady=(8, 4), sticky="ew")
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

        calc = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        calc.grid(row=5, column=0, columnspan=2, padx=16, pady=(4, 16), sticky="ew")
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
            return

        self._rf = None
        self._xgb = None
        self._lstm = None
        self._hybrid_weights = None

        rf_path = os.path.join(MODELS_DIR, f"rf_{stem}.pkl")
        if os.path.exists(rf_path):
            self._rf = joblib.load(rf_path)

        xgb_path = os.path.join(MODELS_DIR, f"xgb_{stem}.pkl")
        if os.path.exists(xgb_path):
            self._xgb = joblib.load(xgb_path)

        self._lstm = _load_lstm(stem)
        self._hybrid_weights = _load_hybrid_weights(stem)

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
            self._field_resetters[key] = lambda s=slider, lbl=val_lbl, idx=default_idx, vals=val_list, n=max(len(val_list)-1,1), curr=current_idx: (
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

    def _on_model_change(self, _choice: str) -> None:
        if not self._all_preds:
            return
        selected = self._model_var.get()
        pred = self._all_preds.get(selected)
        if pred is not None:
            self._result_label.configure(text=f"{pred:.2f}")
            self._unit_label.configure(text=self._unit_label_text(selected))
            self._update_calculator()
        self._draw_chart(self._all_preds, selected)

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
        unit = get_effort_unit(stem)
        self._effort_unit = unit

        def predict(model_name: str) -> float | None:
            try:
                if model_name == "SVR":
                    if self._svr is None:
                        return None
                    return to_person_months(float(self._svr.predict(X_scaled)[0]), unit)
                if model_name == "Linear Regression":
                    if self._lr is None:
                        return None
                    return to_person_months(float(self._lr.predict(X_scaled)[0]), unit)
                if model_name == "Random Forest":
                    if self._rf is None:
                        return None
                    return to_person_months(float(self._rf.predict(X_scaled)[0]), unit)
                if model_name == "XGBoost":
                    if self._xgb is None:
                        return None
                    return to_person_months(float(self._xgb.predict(X_scaled)[0]), unit)
                if model_name == "LSTM":
                    if self._lstm is None:
                        return None
                    xt = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
                    with torch.no_grad():
                        raw = float(self._lstm(xt).item())
                    return to_person_months(raw, unit)
                if model_name == "Hybrid":
                    if self._hybrid_weights is None or self._lstm is None or self._xgb is None or self._lr is None:
                        return None
                    p_lstm = predict("LSTM")
                    p_xgb = predict("XGBoost")
                    p_lr = predict("Linear Regression")
                    if p_lstm is None or p_xgb is None or p_lr is None:
                        return None
                    w = self._hybrid_weights
                    return w["lstm"] * p_lstm + w["xgb"] * p_xgb + w["lr"] * p_lr
            except Exception:
                return None
            return None

        self._all_preds = {}
        for name in MODEL_NAMES:
            p = predict(name)
            if p is not None:
                self._all_preds[name] = p

        selected = self._model_var.get()
        pred = self._all_preds.get(selected)
        if pred is None:
            self._error_label.configure(
                text=f"Model '{selected}' tidak tersedia untuk dataset ini."
            )
            # Show first available
            for name in MODEL_NAMES:
                if name in self._all_preds:
                    pred = self._all_preds[name]
                    selected = name
                    break

        if pred is not None:
            self._result_label.configure(text=f"{pred:.2f}")
            self._unit_label.configure(text=self._unit_label_text(selected))
        self._draw_chart(self._all_preds, self._model_var.get())
        self._update_calculator()

    def _on_reset(self) -> None:
        for resetter in self._field_resetters.values():
            resetter()
        self._all_preds = {}
        if hasattr(self, "_error_label"):
            self._error_label.configure(text="")
        if hasattr(self, "_result_label"):
            self._result_label.configure(text="—")
        if hasattr(self, "_unit_label"):
            self._unit_label.configure(text="")
        if hasattr(self, "_model_var"):
            self._model_var.set("SVR")
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
        if not self._all_preds:
            return
        selected = self._model_var.get()
        pred = self._all_preds.get(selected)
        if pred is None:
            return
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

    def _draw_chart(self, all_preds: dict[str, float], selected: str) -> None:
        if not all_preds:
            self._draw_empty_chart()
            return

        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        if self._fig:
            plt.close(self._fig)

        fig, ax = plt.subplots(figsize=(4.2, 5.8), facecolor=CHART_BG)
        ax.set_facecolor(CHART_BG)
        fig.subplots_adjust(left=0.30, right=0.88, top=0.90, bottom=0.08)

        labels = []
        values = []
        colors = []
        for name in MODEL_NAMES:
            if name in all_preds:
                labels.append(name)
                values.append(all_preds[name])
                colors.append(COLOR_HIGHLIGHT if name == selected else COLOR_MUTED)

        if not values:
            self._draw_empty_chart()
            return

        max_val = max(values) if values else 1
        bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="none")
        for bar, val in zip(bars, values):
            ax.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center", ha="left",
                color=CHART_FG, fontsize=9, fontweight="bold",
            )

        xlabel = (
            f"Person-Months (÷{HOURS_PER_MONTH})"
            if self._effort_unit == "person-hours"
            else "Person-Months"
        )
        ax.set_xlabel(xlabel, color="#b3b3b3", fontsize=8)
        ax.set_title("Prediksi Effort", color=CHART_FG, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="#b3b3b3", labelsize=9)
        ax.set_xlim(0, max_val * 1.40 + 1)
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
