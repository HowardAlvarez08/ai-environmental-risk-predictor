import os
import joblib
import pandas as pd
import streamlit as st


# -------------------------------------------------
# Load all trained models (cached by Streamlit)
# -------------------------------------------------
@st.cache_resource
def load_models(models_dir="models"):
    models = {}

    model_files = {
        "flood": "flood_model.pkl",
        "storm": "storm_model.pkl",
        "rain": "rain_model.pkl",
        "landslide": "landslide_model.pkl",
    }

    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        models[name] = joblib.load(path)

    return models


# -------------------------------------------------
# Predict risk probabilities
# -------------------------------------------------
def predict_risks(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Takes engineered feature DataFrame and trained models
    Returns DataFrame with risk probability columns appended
    """

    df_out = df.copy()

    # -------------------------------------------------
    # IMPORTANT FIX:
    # Drop all non-numeric columns (timestamps, objects)
    # -------------------------------------------------
    X = df_out.select_dtypes(include=["number"])

    # Optional safety check
    if X.empty:
        raise ValueError("No numeric features available for prediction.")

    # -------------------------------------------------
    # Predict probabilities
    # -------------------------------------------------
    df_out["flood_risk_prob"] = models["flood"].predict_proba(X)[:, 1]
    df_out["storm_risk_prob"] = models["storm"].predict_proba(X)[:, 1]
    df_out["rain_risk_prob"] = models["rain"].predict_proba(X)[:, 1]
    df_out["landslide_risk_prob"] = models["landslide"].predict_proba(X)[:, 1]

    return df_out
