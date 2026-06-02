## 1. Project Scaffold

- [x] 1.1 Create `requirements.txt` with pinned versions: customtkinter, scikit-learn, pandas, numpy, joblib
- [x] 1.2 Create `data/` directory (add `.gitkeep` so it's tracked by git)
- [x] 1.3 Create `models/` directory (add `.gitkeep` so it's tracked by git)

## 2. Model Training Pipeline (`model_pipeline.py`)

- [x] 2.1 Implement `load_or_generate_data()` — loads `data/dataset.csv` if present, else generates synthetic COCOMO81 dummy data and prints a WARNING
- [x] 2.2 Implement `preprocess(df)` — separates features (KLOC, reliability, complexity, team_size, schedule) and target (effort), applies `StandardScaler`, returns X_train, X_test, y_train, y_test, scaler
- [x] 2.3 Implement `train_models(X_train, y_train)` — trains `SVR(kernel='rbf')` and `LinearRegression`, returns both fitted model objects
- [x] 2.4 Implement `evaluate(models, X_test, y_test)` — computes and prints MAE and R² for each model
- [x] 2.5 Implement `export_artifacts(svr, lr, scaler)` — creates `models/` if needed, serializes all three to `.pkl` with `joblib.dump`
- [x] 2.6 Wire up `if __name__ == "__main__"` block to call functions in order and confirm successful export via stdout

## 3. Desktop GUI Application (`app.py`)

- [x] 3.1 Set `customtkinter.set_appearance_mode("dark")` and `set_default_color_theme("blue")` at module level before any window creation
- [x] 3.2 Implement `load_models()` — loads `svr_model.pkl`, `lr_model.pkl`, `scaler.pkl`; returns `None` and error message string on `FileNotFoundError`
- [x] 3.3 Build main `CTk` window with title, fixed or responsive size, and dark `CTkFrame` container using `grid` layout
- [x] 3.4 Add labeled `CTkEntry` input fields for: KLOC, Reliability multiplier, Complexity multiplier, Team Size, Schedule
- [x] 3.5 Add `CTkComboBox` or `CTkSegmentedButton` for model selection (SVR / Linear Regression)
- [x] 3.6 Add "Estimasi" `CTkButton` that triggers prediction; add "Reset Form" `CTkButton` that clears all fields and output
- [x] 3.7 Implement `on_estimate()` — validates all fields are numeric (show inline `CTkLabel` error if not), scales inputs with scaler, runs selected model, displays result in output label
- [x] 3.8 Implement `on_reset()` — clears all `CTkEntry` fields and the output label
- [x] 3.9 Handle missing model files at startup: show `CTkMessagebox` (or fallback `CTkLabel`) with instruction to run `model_pipeline.py`, disable "Estimasi" button
- [x] 3.10 Wire up `if __name__ == "__main__"` block; call `load_models()` and launch `app.mainloop()`

## 4. Verification

- [x] 4.1 Run `model_pipeline.py` with no CSV present — confirm WARNING printed, dummy data used, three `.pkl` files created in `models/`
- [x] 4.2 Run `model_pipeline.py` with a real or synthetic CSV — confirm it loads and trains correctly
- [x] 4.3 Launch `app.py` — confirm dark theme, all input fields render, model selector works
- [x] 4.4 Enter valid numeric inputs, click "Estimasi" — confirm a numeric prediction appears
- [x] 4.5 Enter a non-numeric value, click "Estimasi" — confirm inline error message and no crash
- [x] 4.6 Click "Reset Form" — confirm all fields and output are cleared
- [x] 4.7 Delete `models/` and launch `app.py` — confirm graceful error message and disabled "Estimasi" button

## 5. UI Enhancements

- [x] 5.1 Tambahkan field `hint` pada definisi `FIELDS` — teks keterangan singkat untuk setiap field input (contoh nilai, rentang, satuan)
- [x] 5.2 Render hint sebagai `CTkLabel` kecil berwarna abu-abu di bawah setiap `CTkEntry`
- [x] 5.3 Tambahkan `matplotlib` ke `requirements.txt` dan import `FigureCanvasTkAgg`
- [x] 5.4 Perlebar window dan bagi layout menjadi dua panel: kiri (form) dan kanan (visualisasi)
- [x] 5.5 Buat panel visualisasi dengan `CTkFrame` dan embed `matplotlib` canvas
- [x] 5.6 Setelah "Estimasi" ditekan, render bar chart perbandingan prediksi SVR vs Linear Regression
- [x] 5.7 Pastikan chart di-reset (cleared) saat "Reset Form" ditekan