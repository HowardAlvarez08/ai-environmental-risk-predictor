# app.py

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz

from src.data_fetch import fetch_real_time_weather
from src.feature_engineering import engineer_features
from src.predict import load_models, predict_risks
from src.recommendation import apply_risk_alerts

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Environmental Risk Predictor",
    layout="wide"
)

st.title("🌏 AI Environmental Risk Predictor")
st.write("Real-time weather-driven risk assessment for floods, storms, rain, and landslides.")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Location & Forecast Settings")

latitude = st.sidebar.number_input("Latitude", value=14.5995)
longitude = st.sidebar.number_input("Longitude", value=120.9842)
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=7, value=1)

refresh = st.sidebar.button("🔄 Fetch & Predict")

# -------------------------------
# Load models once
# -------------------------------
@st.cache_resource
def load_all_models():
    return load_models("models")

models = load_all_models()

# -------------------------------
# Main Pipeline
# -------------------------------
if refresh:
    with st.spinner("Fetching real-time weather data..."):
        df_raw = fetch_real_time_weather(latitude, longitude, forecast_days=forecast_days)

    st.success("✅ Weather data fetched")

    with st.spinner("Engineering features..."):
        df_features = engineer_features(df_raw)

    with st.spinner("Predicting risks..."):
        df_pred = predict_risks(df_features, models)

    with st.spinner("Applying recommendations..."):
        df_final = apply_risk_alerts(df_pred)

    # -------------------------------
    # Current Hour Risk Status
    # -------------------------------
    st.subheader("⏰ Current Hour Risk Status")
    manila_tz = pytz.timezone("Asia/Manila")
    now_ph = datetime.now(manila_tz)
    now_ph_str = now_ph.strftime("%I:%M %p")
    st.write(f"Current Time: {now_ph_str}")

    # Find the row closest to current hour
    df_final['hour_diff'] = abs(df_final['date'] - now_ph)
    current_row = df_final.loc[df_final['hour_diff'].idxmin()]

    cols_current = st.columns(4)
    risk_columns = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]

    for i, risk in enumerate(risk_columns):
        cols_current[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{current_row[risk + '_risk_prob']:.2f}",
            delta=current_row.get(risk + "_alert", 0)
        )

    df_final.drop(columns=['hour_diff'], inplace=True)

    # -------------------------------
    # Display Latest Risk Assessment
    # -------------------------------
    st.subheader("📊 Latest Risk Assessment")
    latest = df_final.iloc[-1]

    cols = st.columns(4)
    for i, risk in enumerate(risk_columns):
        cols[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{latest[risk + '_risk_prob']:.2f}",
            delta=latest.get(risk + "_alert", 0)
        )

    # -------------------------------
    # Detailed Output
    # -------------------------------
    st.subheader("🧾 Detailed Output")

    # Only relevant columns
    feature_cols = [c for c in df_final.columns if "risk_prob" in c or "alert" in c or c in df_raw.columns]
    display_cols = ["date"] + feature_cols

    display_df = df_final[display_cols].tail(24)
    display_df = display_df.loc[:, ~display_df.columns.duplicated()]  # remove duplicate columns

    st.dataframe(display_df, use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
