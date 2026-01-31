# src/predict.py

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load_models(model_dir: str = "models"):
    """
    Load all trained risk models from the models directory.
    """
    model_dir = Path(model_dir)

    models = {}
    for pkl_file in model_dir.glob("*.pkl"):
        model_name = pkl_file.stem
        with open(pkl_file, "rb") as f:
            models[model_name] = pickle.load(f)

    return models


def predict_risks(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Predict risk probabilities and classes using trained models.
    """
    df = df.copy()

    for risk_name, model in models.items():
        try:
            # Predict probabilities (safe for single-class models)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)

                if proba.shape[1] == 2:
                    df[f"{risk_name}_prob"] = proba[:, 1]
                else:
                    df[f"{risk_name}_prob"] = (
                        np.ones(len(df))
                        if model.classes_[0] == 1
                        else np.zeros(len(df))
                    )
            else:
                df[f"{risk_name}_prob"] = model.predict(df)

            # Predict class
            df[f"{risk_name}_pred"] = model.predict(df)

        except Exception as e:
            df[f"{risk_name}_prob"] = np.nan
            df[f"{risk_name}_pred"] = np.nan
            print(f"⚠️ Prediction failed for {risk_name}: {e}")

    return df
