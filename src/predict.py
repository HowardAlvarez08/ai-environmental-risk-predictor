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
    """
    Predict risk probabilities and classes using trained models.
    Handles single-class models safely.
    """
    df_out = df.copy()

    # Use only numeric columns (drop datetime, strings, etc.)
    X = df.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("No numeric features available for prediction.")

    for risk_name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)

                # ✅ Case 1: normal binary model
                if proba.shape[1] == 2:
                    df_out[f"{risk_name}_risk_prob"] = proba[:, 1]

                # ✅ Case 2: single-class model
                else:
                    only_class = model.classes_[0]
                    df_out[f"{risk_name}_risk_prob"] = (
                        np.ones(len(X)) if only_class == 1 else np.zeros(len(X))
                    )
            else:
                # Fallback (rare)
                df_out[f"{risk_name}_risk_prob"] = model.predict(X)

            # Predicted class
            df_out[f"{risk_name}_risk_pred"] = model.predict(X)

        except Exception as e:
            df_out[f"{risk_name}_risk_prob"] = np.nan
            df_out[f"{risk_name}_risk_pred"] = np.nan
            print(f"⚠️ Prediction failed for {risk_name}: {e}")

    return df_out
