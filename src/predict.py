# src/predict.py

import numpy as np
import pandas as pd

def align_features(X, trained_features, clip_ranges=None):
    """
    Ensure runtime features match training features exactly.
    clip_ranges: dict of (min, max) per feature to avoid unrealistic inputs
    """
    # Add missing columns
    for col in trained_features:
        if col not in X.columns:
            X[col] = 0

    # Drop extra columns
    X = X[trained_features]

    # Clip features to expected ranges (prevents weird forecast values)
    if clip_ranges:
        for col, (min_val, max_val) in clip_ranges.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=min_val, upper=max_val)

    return X


def predict_risks(df: pd.DataFrame, models: dict, clip_ranges=None) -> pd.DataFrame:
    """
    Predict risk probabilities and classes using trained models.
    Handles single-class models safely.
    clip_ranges: dict of (min, max) to sanitize input features
    """
    df_out = df.copy()
    X = df_out.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("No numeric features available for prediction.")

    for risk_name, model in models.items():
        try:
            # Align features per model
            trained_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else X.columns
            X_model = align_features(X, trained_features, clip_ranges)

            # Prediction
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_model)
                if proba.shape[1] == 2:
                    df_out[f"{risk_name}_risk_prob"] = proba[:, 1]
                else:
                    only_class = model.classes_[0]
                    df_out[f"{risk_name}_risk_prob"] = np.ones(len(X_model)) if only_class == 1 else np.zeros(len(X_model))
            else:
                df_out[f"{risk_name}_risk_prob"] = model.predict(X_model)

            df_out[f"{risk_name}_risk_pred"] = model.predict(X_model)

        except Exception as e:
            df_out[f"{risk_name}_risk_prob"] = np.nan
            df_out[f"{risk_name}_risk_pred"] = np.nan
            print(f"⚠️ Prediction failed for {risk_name}: {e}")

    return df_out
