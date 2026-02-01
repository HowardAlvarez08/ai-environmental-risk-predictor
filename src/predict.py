# src/predict.py

import os
import joblib
import pandas as pd
import numpy as np


def load_models(model_dir="models"):
    """Load all risk models from Joblib files"""
    models = {}

    model_files = {
        "flood": "flood_model.joblib",
        "rain": "rain_model.joblib",
        "storm": "storm_model.joblib",
        "landslide": "landslide_model.joblib",
    }

    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"⚠️ Model file not found: {path}")

    return models


def align_features(X: pd.DataFrame, trained_features: list) -> pd.DataFrame:
    """
    Align runtime features to the trained features.
    Adds missing columns with zeros and drops extra columns.
    """
    # Add missing columns
    for col in trained_features:
        if col not in X.columns:
            X[col] = 0

    # Drop extra columns
    X = X[trained_features]

    return X


def predict_risks(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Predict risk probabilities and classes using trained models.
    Handles missing numeric columns, single-class models, and missing feature info.
    """
    df_out = df.copy()

    # Use only numeric columns
    X = df.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("No numeric features available for prediction.")

    for risk_name, model in models.items():
        try:
            # Try to get trained feature names, fallback to runtime features
            trained_features = getattr(model, "feature_names_in_", X.columns.tolist())
            X_aligned = align_features(X, trained_features)

            # Predict probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_aligned)

                # Binary model
                if proba.shape[1] == 2:
                    df_out[f"{risk_name}_risk_prob"] = proba[:, 1]
                # Single-class model
                else:
                    only_class = model.classes_[0]
                    df_out[f"{risk_name}_risk_prob"] = np.ones(len(X)) if only_class == 1 else np.zeros(len(X))
            else:
                # Fallback prediction
                df_out[f"{risk_name}_risk_prob"] = model.predict(X_aligned)

            # Predicted class
            df_out[f"{risk_name}_risk_pred"] = model.predict(X_aligned)

        except Exception as e:
            df_out[f"{risk_name}_risk_prob"] = np.nan
            df_out[f"{risk_name}_risk_pred"] = np.nan
            print(f"⚠️ Prediction failed for {risk_name}: {e}")

    return df_out
