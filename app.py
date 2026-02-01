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
st.sidebar.header("Location Settings")

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
# Recommended actions mapping
# -------------------------------
risk_actions = {
    "Low": "✅ Normal conditions. Stay informed and follow usual precautions.",
    "Moderate": "⚠️ Moderate risk. Prepare emergency kits, secure property, monitor weather updates.",
    "Severe": "🚨 High risk. Follow local authorities, evacuate if necessary, avoid travel."
}

def risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "Severe"

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
    # Ensure PH timezone
    # -------------------------------
    now_ph = datetime.now(pytz.timezone("Asia/Manila"))

    if df_final["date"].dt.tz is None:
        df_final["date"] = df_final["date"].dt.tz_localize("Asia/Manila")

    # -------------------------------
    # Forecast Hour Selector
    # -------------------------------
    forecast_options = {
        "Now": 0,
        "+1 hour": 1,
        "+2 hours": 2,
        "+4 hours": 4,
        "+8 hours": 8,
        "+12 hours": 12,
        "+16 hours": 16,
        "+24 hours": 24
    }
    selected_forecast = st.selectbox("📅 Choose Forecast Time", list(forecast_options.keys()))
    selected_hours = forecast_options[selected_forecast]

    target_time = now_ph + timedelta(hours=selected_hours)
    target_time_str = target_time.strftime("%I:%M %p")

    st.subheader(f"⏱ Risk Status for {selected_forecast} ({target_time_str})")

    # -------------------------------
    # Find the row closest to target time
    # -------------------------------
    df_final['hour_diff'] = abs(df_final['date'] - target_time)
    current_row = df_final.loc[df_final['hour_diff'].idxmin()]

    # -------------------------------
    # Current Hour Dashboard
    # -------------------------------
    risk_columns = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]
    cols_current = st.columns(len(risk_columns))
    for i, risk in enumerate(risk_columns):
        prob = current_row[risk + "_risk_prob"]
        level = risk_level(prob)
        action = risk_actions[level]
        alert = current_row.get(risk + "_risk_alert", "")

        if level == "Low":
            color = "✅"
        elif level == "Moderate":
            color = "⚠️"
        else:
            color = "🚨"

        cols_current[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{prob:.2f}",
            delta=f"{color} {alert}"
        )

    # -------------------------------
    # Recommended Actions Section
    # -------------------------------
    st.subheader("🛠 Recommended Actions / Preparations")
    for risk in risk_columns:
        prob = current_row[risk + "_risk_prob"]
        level = risk_level(prob)
        st.markdown(f"**{risk.replace('_', ' ').title()} ({level} Risk)**: {risk_actions[level]}")

    # -------------------------------
    # Simplified Detailed Output Table
    # -------------------------------
    feature_cols = ["date"] + [c for c in df_final.columns if "risk_prob" in c or "risk_alert" in c]
    st.subheader("🧾 Detailed Forecast Data (key columns)")
    st.dataframe(df_final[feature_cols].tail(24), use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
