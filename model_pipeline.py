import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_PATH = os.path.join("data", "dataset.csv")
MODELS_DIR = "models"

FEATURE_COLS = ["kloc", "reliability", "complexity", "team_size", "schedule"]
TARGET_COL = "effort"


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[INFO] Loaded dataset from '{DATA_PATH}' ({len(df)} rows).")
        return df

    print(
        "\nWARNING: 'data/dataset.csv' not found. "
        "Using synthetic COCOMO81 dummy data. "
        "Replace with real data for meaningful predictions.\n"
    )
    rng = np.random.default_rng(42)
    n = 120
    kloc = rng.uniform(1, 500, n)
    reliability = rng.choice([0.75, 0.88, 1.0, 1.15, 1.40], n)
    complexity = rng.choice([0.70, 0.85, 1.0, 1.15, 1.30, 1.65], n)
    team_size = rng.integers(2, 20, n).astype(float)
    schedule = rng.choice([1.10, 1.00, 1.00, 0.91, 0.85], n)
    # Approximate COCOMO intermediate formula: effort = a * KLOC^b * EAF
    eaf = reliability * complexity * schedule
    effort = 3.2 * (kloc ** 1.05) * eaf + rng.normal(0, 5, n)
    effort = np.clip(effort, 1, None)

    return pd.DataFrame(
        {
            "kloc": kloc,
            "reliability": reliability,
            "complexity": complexity,
            "team_size": team_size,
            "schedule": schedule,
            "effort": effort,
        }
    )


def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_models(X_train, y_train):
    svr = SVR(kernel="rbf")
    svr.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return svr, lr


def evaluate(models: dict, X_test, y_test) -> None:
    print("\n--- Model Evaluation (Test Set) ---")
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"  {name:20s}  MAE={mae:.2f}  R2={r2:.4f}")
    print()


def export_artifacts(svr, lr, scaler) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(svr, os.path.join(MODELS_DIR, "svr_model.pkl"))
    joblib.dump(lr, os.path.join(MODELS_DIR, "lr_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"[INFO] Artifacts saved to '{MODELS_DIR}/'.")
    print("       svr_model.pkl | lr_model.pkl | scaler.pkl")


if __name__ == "__main__":
    df = load_or_generate_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    svr, lr = train_models(X_train, y_train)
    evaluate({"SVR": svr, "LinearRegression": lr}, X_test, y_test)
    export_artifacts(svr, lr, scaler)
    print("\n[OK] Pipeline complete. You can now launch app.py.")
