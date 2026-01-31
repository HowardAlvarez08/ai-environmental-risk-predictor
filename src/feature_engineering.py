# src/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()

    if "date" not in df.columns:
        raise KeyError("Missing 'date' column in dataframe")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Time-based features
    df['month'] = df['date'].dt.month
    df['is_habagat'] = df['month'].isin([6,7,8,9]).astype(int)
    df['is_amihan'] = df['month'].isin([11,12,1,2]).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df["month"] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    lag_hours = [1,6,24]
    for lag in lag_hours:
        df[f'precip_lag{lag}h'] = df['precipitation_sum'].shift(lag).fillna(0)
        df[f'soil_lag{lag}h'] = df['soil_moisture_mean'].shift(lag).fillna(0)

    # Rolling sums
    df['precip_6h_sum'] = df['precipitation_sum'].rolling(6, min_periods=1).sum()
    df['precip_12h_sum'] = df['precipitation_sum'].rolling(12, min_periods=1).sum()
    df['precip_24h_sum'] = df['precipitation_sum'].rolling(24, min_periods=1).sum()
    df['precip_24h_mean'] = df['precipitation_sum'].rolling(24, min_periods=1).mean()

    # Thresholds
    df['heavy_rain'] = (df['precipitation_sum'] >= 7.5).astype(int)
    df['very_heavy_rain'] = (df['precipitation_sum'] >= 15.0).astype(int)

    # Interaction
    df['rain_soil_interaction'] = df['precipitation_sum'] * df['soil_moisture_mean']

    # Fill remaining NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    df = df.set_index('date').sort_index()
    return df
