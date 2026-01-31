# src/data_fetch.py
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests

def fetch_real_time_weather(lat=14.5995, lon=120.9842, timezone="Asia/Manila"):
    # -------------------------------
    # API Setup
    # -------------------------------
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m","relative_humidity_2m","dew_point_2m","pressure_msl",
            "cloud_cover","wind_speed_10m","wind_gusts_10m","precipitation","rain",
            "soil_moisture_0_to_1cm","soil_moisture_1_to_3cm","soil_moisture_3_to_9cm",
            "soil_moisture_9_to_27cm","soil_moisture_27_to_81cm",
            "soil_temperature_0cm","soil_temperature_6cm","soil_temperature_18cm",
            "soil_temperature_54cm"
        ],
        "timezone": "Asia/Singapore"
    }

    responses = client.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    # Extract variables dynamically
    for i, var in enumerate(hourly.Variables()):
        hourly_data[var.Name()] = var.ValuesAsNumpy()

    df = pd.DataFrame(hourly_data)
    
    # -------------------------------
    # Timezone conversion
    # -------------------------------
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(timezone).dt.tz_localize(None)

    # -------------------------------
    # Handle NaNs
    # -------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill()
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    return df
