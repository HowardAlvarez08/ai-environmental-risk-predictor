# src/data_fetch.py

import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests


def fetch_real_time_hourly(
    latitude=14.5995,
    longitude=120.9842,
    timezone="Asia/Singapore"
):
    """
    Fetch real-time hourly weather data from Open-Meteo
    and return a cleaned pandas DataFrame.
    """

    # -------------------------------
    # API setup
    # -------------------------------
    cache_session = requests_cache.CachedSession(
        ".cache", expire_after=3600
    )
    retry_session = retry(
        cache_session, retries=5, backoff_factor=0.2
    )
    openmeteo = openmeteo_requests.Client(
        session=retry_session
    )

    # -------------------------------
    # API request
    # -------------------------------
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "pressure_msl",
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
            "precipitation",
            "rain",
            "soil_moisture_0_to_1cm",
            "soil_moisture_1_to_3cm",
            "soil_moisture_3_to_9cm",
            "soil_moisture_9_to_27cm",
            "soil_moisture_27_to_81cm",
            "soil_temperature_0cm",
            "soil_temperature_6cm",
            "soil_temperature_18cm",
            "soil_temperature_54cm"
        ],
        "timezone": timezone
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    # -------------------------------
    # Extract hourly data
    # -------------------------------
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "dew_point_2m": hourly.Variables(2).ValuesAsNumpy(),
        "pressure_msl": hourly.Variables(3).ValuesAsNumpy(),
        "cloud_cover": hourly.Variables(4).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(5).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(6).ValuesAsNumpy(),
        "precipitation": hourly.Variables(7).ValuesAsNumpy(),
        "rain": hourly.Variables(8).ValuesAsNumpy(),
        "soil_moisture_0_to_1cm": hourly.Variables(9).ValuesAsNumpy(),
        "soil_moisture_1_to_3cm": hourly.Variables(10).ValuesAsNumpy(),
        "soil_moisture_3_to_9cm": hourly.Variables(11).ValuesAsNumpy(),
        "soil_moisture_9_to_27cm": hourly.Variables(12).ValuesAsNumpy(),
        "soil_moisture_27_to_81cm": hourly.Variables(13).ValuesAsNumpy(),
        "soil_temperature_0cm": hourly.Variables(14).ValuesAsNumpy(),
        "soil_temperature_6cm": hourly.Variables(15).ValuesAsNumpy(),
        "soil_temperature_18cm": hourly.Variables(16).ValuesAsNumpy(),
        "soil_temperature_54cm": hourly.Variables(17).ValuesAsNumpy()
    }

    df = pd.DataFrame(hourly_data)

    # -------------------------------
    # Convert timezone → Asia/Manila
    # -------------------------------
    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.tz_convert("Asia/Manila")
        .dt.tz_localize(None)
    )

    # -------------------------------
    # Rename columns (model-compatible)
    # -------------------------------
    df.rename(columns={
        "temperature_2m": "temperature_mean",
        "relative_humidity_2m": "relative_humidity_mean",
        "pressure_msl": "sea_level_pressure_mean",
        "soil_moisture_27_to_81cm": "soil_moisture_mean",
        "wind_speed_10m": "wind_speed_max",
        "wind_gusts_10m": "wind_gust_max",
        "precipitation": "precipitation_sum",
        "rain": "rain_sum",
        "cloud_cover": "cloud_cover_total",
        "dew_point_2m": "dew_point_mean",
        "soil_moisture_0_to_1cm": "soil_moisture_top",
        "soil_moisture_1_to_3cm": "soil_moisture_1_3cm",
        "soil_moisture_3_to_9cm": "soil_moisture_3_9cm",
        "soil_moisture_9_to_27cm": "soil_moisture_9_27cm",
        "soil_temperature_0cm": "soil_temperature_top",
        "soil_temperature_6cm": "soil_temperature_6cm",
        "soil_temperature_18cm": "soil_temperature_18cm",
        "soil_temperature_54cm": "soil_temperature_54cm"
    }, inplace=True)

    # -------------------------------
    # Handle NaNs safely
    # -------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )
    df[numeric_cols] = df[numeric_cols].ffill()

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    return df
