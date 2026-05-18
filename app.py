import os

import customtkinter as ctk
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Theme (must be set before any window is created) ──────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODELS_DIR = "models"

# (label, key, hint)
FIELDS = [
    (
        "KLOC",
        "kloc",
        "Ribuan baris kode. Contoh: 25 = 25.000 baris. Rentang: 1 – 500",
    ),
    (
        "Reliability Multiplier",
        "reliability",
        "Dampak kegagalan sistem: 0.75 (rendah) | 1.00 (normal) | 1.40 (kritis)",
    ),
    (
        "Complexity Multiplier",
        "complexity",
        "Kompleksitas teknis: 0.70 (sederhana) | 1.00 (normal) | 1.65 (AI/real-time)",
    ),
    (
        "Team Size",
        "team_size",
        "Jumlah developer aktif. Contoh: 4",
    ),
    (
        "Required Schedule",
        "schedule",
        "Faktor jadwal: 0.85 (dipercepat) | 1.00 (normal) | 1.10 (ada kelonggaran)",
    ),
]

CHART_BG = "#1c1c1c"
CHART_FG = "#ececec"
COLOR_SVR = "#61afef"
COLOR_LR = "#98c379"


def load_models():
    try:
        svr = joblib.load(os.path.join(MODELS_DIR, "svr_model.pkl"))
        lr = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        return svr, lr, scaler, None
    except FileNotFoundError as exc:
        return None, None, None, (
            f"File model tidak ditemukan: {exc}\n\n"
            "Jalankan terlebih dahulu:\n  python model_pipeline.py"
        )


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OpenSpec - AI Cost Estimator")
        self.geometry("960x620")
        self.resizable(False, False)

        self.svr, self.lr, self.scaler, err = load_models()
        self._canvas = None
        self._fig = None
        self._build_ui(err)

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self, load_error: str | None) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Left panel: form ──────────────────────────────────────────────────
        left = ctk.CTkFrame(self, corner_radius=12, width=460)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        left.grid_propagate(False)
        left.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            left,
            text="Estimasi Biaya Perangkat Lunak",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(18, 10), padx=20)

        if load_error:
            self._build_error_banner(left, load_error)
        else:
            self._build_form(left)

        # ── Right panel: visualization ────────────────────────────────────────
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
        self._build_empty_chart(right)

    def _build_form(self, parent: ctk.CTkFrame) -> None:
        self._entries: dict[str, ctk.CTkEntry] = {}

        row = 1
        for label_text, key, hint in FIELDS:
            # Field label
            ctk.CTkLabel(
                parent, text=label_text, anchor="w", font=ctk.CTkFont(size=13)
            ).grid(row=row, column=0, padx=(20, 8), pady=(8, 0), sticky="w")

            # Entry
            entry = ctk.CTkEntry(parent, width=160, placeholder_text="0.0")
            entry.grid(
                row=row, column=1, padx=(0, 20), pady=(8, 0), sticky="ew"
            )
            self._entries[key] = entry
            row += 1

            # Hint label spans both columns, sits below the entry row
            ctk.CTkLabel(
                parent,
                text=hint,
                anchor="w",
                text_color="gray70",
                font=ctk.CTkFont(size=10),
                wraplength=390,
            ).grid(
                row=row, column=0, columnspan=2, padx=(20, 20), pady=(0, 2), sticky="w"
            )
            row += 1

        # Model selector
        ctk.CTkLabel(
            parent, text="Model Prediksi", anchor="w", font=ctk.CTkFont(size=13)
        ).grid(row=row, column=0, padx=(20, 8), pady=(10, 0), sticky="w")
        self._model_var = ctk.StringVar(value="SVR")
        ctk.CTkSegmentedButton(
            parent,
            values=["SVR", "Linear Regression"],
            variable=self._model_var,
        ).grid(row=row, column=1, padx=(0, 20), pady=(10, 0), sticky="ew")
        row += 1

        # Inline validation error
        self._error_label = ctk.CTkLabel(
            parent, text="", text_color="#e06c75", font=ctk.CTkFont(size=11)
        )
        self._error_label.grid(
            row=row, column=0, columnspan=2, pady=(6, 0), padx=20
        )
        row += 1

        # Buttons
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.grid(row=row, column=0, columnspan=2, pady=(10, 8))
        ctk.CTkButton(
            btn_frame, text="Estimasi", width=130, command=self._on_estimate
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            btn_frame,
            text="Reset Form",
            width=130,
            fg_color="gray30",
            hover_color="gray40",
            command=self._on_reset,
        ).pack(side="left", padx=6)
        row += 1

        # Result area (at bottom of left panel)
        result_frame = ctk.CTkFrame(parent, fg_color="gray17", corner_radius=8)
        result_frame.grid(
            row=row, column=0, columnspan=2, padx=20, pady=(6, 16), sticky="ew"
        )
        result_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            result_frame,
            text="Hasil Estimasi",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, pady=(10, 2))

        self._result_label = ctk.CTkLabel(
            result_frame,
            text="—",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#61afef",
        )
        self._result_label.grid(row=1, column=0, pady=(2, 4))

        self._unit_label = ctk.CTkLabel(
            result_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        )
        self._unit_label.grid(row=2, column=0, pady=(0, 10))

    def _build_error_banner(self, parent: ctk.CTkFrame, message: str) -> None:
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
            text=message,
            text_color="gray70",
            font=ctk.CTkFont(size=12),
            justify="left",
        ).pack(padx=16, pady=(0, 14))

    # ── Chart ──────────────────────────────────────────────────────────────────

    def _build_empty_chart(self, parent: ctk.CTkFrame) -> None:
        fig, ax = plt.subplots(figsize=(4.2, 4.8), facecolor=CHART_BG)
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
        self._embed_chart(fig, parent)

    def _update_chart(self, pred_svr: float, pred_lr: float) -> None:
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        if self._fig:
            plt.close(self._fig)

        fig, ax = plt.subplots(figsize=(4.2, 4.8), facecolor=CHART_BG)
        ax.set_facecolor(CHART_BG)
        fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.12)

        models = ["SVR", "Linear\nRegression"]
        values = [pred_svr, pred_lr]
        colors = [COLOR_SVR, COLOR_LR]

        bars = ax.barh(models, values, color=colors, height=0.45, edgecolor="none")

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(
                val + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center", ha="left",
                color=CHART_FG, fontsize=11, fontweight="bold",
            )

        ax.set_xlabel("Person-Months", color="#b3b3b3", fontsize=10)
        ax.set_title("Prediksi Effort", color=CHART_FG, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="#b3b3b3", labelsize=10)
        ax.set_xlim(0, max(values) * 1.30 + 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.grid(True, color="#404040", linewidth=0.5)
        ax.set_axisbelow(True)

        self._embed_chart(fig, self._chart_frame)
        self._fig = fig

    def _embed_chart(self, fig, parent: ctk.CTkFrame) -> None:
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=1, column=0, padx=10, pady=(0, 14), sticky="nsew")
        self._canvas = canvas
        self._fig = fig

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_estimate(self) -> None:
        self._error_label.configure(text="")
        values = []
        for label_text, key, _ in FIELDS:
            raw = self._entries[key].get().strip()
            try:
                values.append(float(raw))
            except ValueError:
                self._error_label.configure(
                    text=f"Input tidak valid pada kolom '{label_text}'. Masukkan angka."
                )
                return

        X = np.array(values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        pred_svr = self.svr.predict(X_scaled)[0]
        pred_lr = self.lr.predict(X_scaled)[0]

        selected = self._model_var.get()
        prediction = pred_svr if selected == "SVR" else pred_lr

        self._result_label.configure(text=f"{prediction:.2f}")
        self._unit_label.configure(text=f"Person-Months  [{selected}]")
        self._update_chart(pred_svr, pred_lr)

    def _on_reset(self) -> None:
        for entry in self._entries.values():
            entry.delete(0, "end")
        self._error_label.configure(text="")
        self._result_label.configure(text="—")
        self._unit_label.configure(text="")
        self._build_empty_chart(self._chart_frame)


if __name__ == "__main__":
    app = App()
    app.mainloop()
