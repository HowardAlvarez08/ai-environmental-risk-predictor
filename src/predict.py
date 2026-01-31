import os
import joblib
import pandas as pd


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
        models[name] = joblib.load(path)

    return models


def align_features(X, trained_features):
    """
    Ensure runtime features match training features exactly
    """
    # Add missing columns
    for col in trained_features:
        if col not in X.columns:
            X[col] = 0

    # Drop extra columns
    X = X[trained_features]

    return X


def predict_risks(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    df_out = df.copy()

    # Numeric only
    X = df_out.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("No numeric features available for prediction.")

    # 🔑 ALIGN FEATURES PER MODEL
    for risk in models:
        trained_features = models[risk].feature_names_in_
        X_aligned = align_features(X.copy(), trained_features)

        df_out[f"{risk}_risk_prob"] = models[risk].predict_proba(X_aligned)[:, 1]

    return df_out
