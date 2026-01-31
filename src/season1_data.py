# src/season1_data.py
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
import calendar

def fetch_era5_data(lat=14.5995, lon=120.9842, years=range(2018, 2024)):
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.5)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm",
        "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm",
        "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "pressure_msl", "cloud_cover", "visibility", "cape", "lifted_index"
    ]

    all_frames = []
    for year in years:
        for month in range(1,13):
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
            responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/era5",
                                              {"latitude": lat, "longitude": lon,
                                               "start_date": start_date, "end_date": end_date,
                                               "hourly": variables,
                                               "timezone": "Asia/Singapore"})
            response = responses[0]
            hourly = response.Hourly()
            hourly_data = {"date": pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                                                 end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                                                 freq=pd.Timedelta(seconds=hourly.Interval()),
                                                 inclusive="left")}
            for i, var in enumerate(variables):
                hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
            df_month = pd.DataFrame(hourly_data)
            df_month["date"] = df_month["date"].dt.tz_convert("Asia/Manila").dt.tz_localize(None)
            all_frames.append(df_month)
    df = pd.concat(all_frames, ignore_index=True)
    return df

def preprocess_and_feature_engineer(df):
    # Copy your NaN handling, renaming, lag features, rolling sums, interactions
    # Return X (features) and y (targets)
    ...
    return X, y_flood, y_rain, y_storm, y_landslide
