import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os


def load_models(model_dir="models"):
    models = {}

    models["flood"] = joblib.load(os.path.join(model_dir, "flood_model.joblib"))
    models["storm"] = joblib.load(os.path.join(model_dir, "storm_model.joblib"))
    models["rain"] = joblib.load(os.path.join(model_dir, "rain_model.joblib"))
    models["landslide"] = joblib.load(os.path.join(model_dir, "landslide_model.joblib"))

    models["dl"] = tf.keras.models.load_model(
        os.path.join(model_dir, "dl_model.keras"),
        compile=False
    )

    print("All models loaded successfully ✅")
    return models


def predict_risks(df, models):
    """
    df: engineered feature dataframe
    models: dict of loaded models
    """

    X = df.values

    df_out = df.copy()

    df_out["flood_risk_prob"] = models["flood"].predict_proba(X)[:, 1]
    df_out["storm_risk_prob"] = models["storm"].predict_proba(X)[:, 1]
    df_out["rain_risk_prob"] = models["rain"].predict_proba(X)[:, 1]
    df_out["landslide_risk_prob"] = models["landslide"].predict_proba(X)[:, 1]

    # Optional DL model
    if "dl" in models:
        df_out["dl_risk_prob"] = models["dl"].predict(X, verbose=0).flatten()

    return df_out
