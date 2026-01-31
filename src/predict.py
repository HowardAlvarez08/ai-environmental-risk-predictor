import os
import joblib
import pandas as pd


# --------------------------------------------------
# Load all trained models (joblib)
# --------------------------------------------------
def load_models(model_dir="models"):
    models = {}

    model_files = {
        "flood": "flood_model.joblib",
        "rain": "rain_model.joblib",
        "storm": "storm_model.joblib",
        "landslide": "landslide_model.joblib",
    }

    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")

        models[name] = joblib.load(path)

    return models


# --------------------------------------------------
# Predict risk probabilities
# --------------------------------------------------
def predict_risks(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    df_out = df.copy()

    # ✅ VERY IMPORTANT:
    # Keep ONLY numeric columns (drops Timestamp, datetime, objects)
    X = df_out.select_dtypes(include=["number"])

    # Optional safety check
    if X.empty:
        raise ValueError("❌ No numeric features available for prediction.")

    # --------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------
    df_out["flood_risk_prob"] = models["flood"].predict_proba(X)[:, 1]
    df_out["rain_risk_prob"] = models["rain"].predict_proba(X)[:, 1]
    df_out["storm_risk_prob"] = models["storm"].predict_proba(X)[:, 1]
    df_out["landslide_risk_prob"] = models["landslide"].predict_proba(X)[:, 1]

    return df_out
