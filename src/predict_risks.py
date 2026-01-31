# src/predict_risks.py
import numpy as np
import pandas as pd

def predict_risks(df_features, models, scaler, continuous_features, binary_features=[], cyclical_features=[]):
    X_continuous_scaled = scaler.transform(df_features[continuous_features])
    X_scaled_df = pd.DataFrame(X_continuous_scaled, columns=continuous_features, index=df_features.index)
    X_for_pred = pd.concat([X_scaled_df, df_features[binary_features + cyclical_features]], axis=1)

    # Align columns
    missing = set(models[list(models.keys())[0]].feature_names_in_) - set(X_for_pred.columns)
    for col in missing:
        X_for_pred[col] = 0.0
    X_for_pred = X_for_pred[models[list(models.keys())[0]].feature_names_in_]

    # Predict
    for risk_name, model in models.items():
        proba = model.predict_proba(X_for_pred)
        if proba.shape[1] == 2:
            df_features[f"{risk_name}_prob"] = proba[:,1]
        else:
            df_features[f"{risk_name}_prob"] = np.ones(len(proba)) if model.classes_[0]==1 else np.zeros(len(proba))
        df_features[f"{risk_name}_pred"] = model.predict(X_for_pred)

    return df_features
