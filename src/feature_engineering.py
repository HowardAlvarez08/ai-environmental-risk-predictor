# src/feature_engineering.py

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to real-time weather data
    to match training-time features.
    """

    df = df.copy()

    # -------------------------------
    # Lag features (1h, 3h, 6h)
    # -------------------------------
    lag_features = [
        "temperature_mean",
        "relative_humidity_mean",
        "rain_sum",
        "precipitation_sum",
        "wind_speed_max",
        "wind_gust_max",
        "soil_moisture_mean",
        "sea_level_pressure_mean"
    ]

    for col in lag_features:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)
            df[f"{col}_lag6"] = df[col].shift(6)

    # -------------------------------
    # Rolling statistics (3h, 6h)
    # -------------------------------
    rolling_features = [
        "rain_sum",
        "precipitation_sum",
        "wind_speed_max",
        "wind_gust_max"
    ]

    for col in rolling_features:
        if col in df.columns:
            df[f"{col}_roll3"] = df[col].rolling(3).mean()
            df[f"{col}_roll6"] = df[col].rolling(6).mean()

    # -------------------------------
    # Binary risk indicators
    # -------------------------------
    if "rain_sum" in df.columns:
        df["heavy_rain_flag"] = (df["rain_sum"] > 10).astype(int)

    if "wind_gust_max" in df.columns:
        df["strong_wind_flag"] = (df["wind_gust_max"] > 40).astype(int)

    # -------------------------------
    # Final NaN handling (post-lags)
    # -------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method="bfill")
    df[numeric_cols] = df[numeric_cols].fillna(method="ffill")
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df
