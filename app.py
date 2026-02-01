# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
# Clip ranges to prevent unrealistic predictions
# -------------------------------
clip_ranges = {
    "temperature_mean": (15, 40),
    "relative_humidity_mean": (20, 100),
    "wind_speed_max": (0, 25),
    "wind_gust_max": (0, 50),
    "precipitation_sum": (0, 200),
    "rain_sum": (0, 200),
    "cloud_cover_total": (0, 100),
    "soil_moisture_mean": (0, 0.5),
}

# -------------------------------
# Main Pipeline
# -------------------------------
if refresh:
    # Fetch Weather
    with st.spinner("Fetching real-time weather data..."):
        df_raw = fetch_real_time_weather(latitude, longitude, forecast_days=forecast_days)
    st.success("✅ Weather data fetched")

    # Engineer Features
    with st.spinner("Engineering features..."):
        df_features = engineer_features(df_raw)

    # Predict Risks
    with st.spinner("Predicting risks..."):
        df_pred = predict_risks(df_features, models, clip_ranges=clip_ranges)

    # Apply Alerts / Recommendations
    with st.spinner("Applying recommendations..."):
        df_final = apply_risk_alerts(df_pred)

    # -------------------------------
    # Time Selection
    # -------------------------------
    tz_ph = pytz.timezone("Asia/Manila")
    now_ph = datetime.now(tz_ph)
    now_ph_str = now_ph.strftime("%I:%M %p")
    st.write(f"🕒 Current Time (PH): {now_ph_str}")

    # Future hour selector
    future_options = [0, 1, 2, 4, 8, 12, 16, 24]  # hours from now
    future_hours = st.selectbox(
        "Select hour from now to view risk status:",
        future_options,
        format_func=lambda x: f"+{x} hour(s)" if x > 0 else "Now"
    )

    target_time = now_ph + timedelta(hours=future_hours)

    # Ensure df_final['date'] is tz-aware
    if df_final['date'].dt.tz is None:
        df_final['date'] = df_final['date'].dt.tz_localize(tz_ph)

    # Find closest hour row
    df_final['hour_diff'] = abs(df_final['date'] - target_time)
    current_row = df_final.loc[df_final['hour_diff'].idxmin()]

    # -------------------------------
    # Current / Future Hour Dashboard
    # -------------------------------
    st.subheader(f"📊 Risk Status at {target_time.strftime('%I:%M %p')} (PH)")

    cols_current = st.columns(4)
    risk_cols = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]

    for i, risk in enumerate(risk_cols):
        prob = current_row[risk + "_risk_prob"]
        alert = current_row[risk + "_risk_alert"]
        cols_current[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{prob:.2f}",
            delta=alert
        )

    # -------------------------------
    # Recommended Actions
    # -------------------------------
    st.subheader("🛠 Recommended Actions / Preparations")

    def recommend_actions(prob):
        """Return action based on risk probability"""
        if prob < 0.3:
            return "Low Risk: Stay alert, monitor local weather updates."
        elif prob < 0.7:
            return "Moderate Risk: Prepare emergency kit, review evacuation plans."
        else:
            return "Severe Risk: Follow official advisories immediately, consider evacuation."

    for risk in risk_cols:
        prob = current_row[risk + "_risk_prob"]
        action = recommend_actions(prob)
        st.markdown(f"**{risk.replace('_',' ').title()}** ({prob:.2f}): {action}")

    # -------------------------------
    # Simplified Detailed Output
    # -------------------------------
    st.subheader("🧾 Detailed Forecast Output (Relevant Columns)")
    feature_cols = [c for c in df_final.columns if "risk_prob" in c or "risk_alert" in c]
    display_cols = ["date"] + feature_cols
    st.dataframe(df_final[display_cols].tail(24), use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
