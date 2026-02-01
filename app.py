# app.py

import streamlit as st
import pandas as pd
import numpy as np
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
st.sidebar.header("Location Settings")
latitude = st.sidebar.number_input("Latitude", value=14.5995)
longitude = st.sidebar.number_input("Longitude", value=120.9842)

st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider(
    "Forecast Days (1–7)", min_value=1, max_value=7, value=1
)

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
    # -------------------------------
    # Fetch weather
    # -------------------------------
    with st.spinner("Fetching real-time weather data..."):
        df_raw = fetch_real_time_weather(latitude, longitude, forecast_days=forecast_days)
    st.success("✅ Weather data fetched")

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    with st.spinner("Engineering features..."):
        df_features = engineer_features(df_raw)

    # -------------------------------
    # Predict Risks
    # -------------------------------
    with st.spinner("Predicting risks..."):
        df_pred = predict_risks(df_features, models)

    # -------------------------------
    # Apply Risk Alerts
    # -------------------------------
    with st.spinner("Applying recommended actions..."):
        df_final = apply_risk_alerts(df_pred)

    # -------------------------------
    # Current Time Dashboard
    # -------------------------------
    now_ph = datetime.now(pytz.timezone("Asia/Manila"))
    now_ph_hour = now_ph.replace(minute=0, second=0, microsecond=0)
    now_ph_str = now_ph.strftime("%I:%M %p")  # AM/PM format
    st.subheader(f"⏱ Current Hour Risk Status ({now_ph_str})")

    # Align datetime types
    if df_final["date"].dt.tz is not None:
        df_final["date"] = df_final["date"].dt.tz_localize(None)

    df_final["hour_diff"] = abs(df_final["date"] - now_ph_hour)
    current_row = df_final.loc[df_final["hour_diff"].idxmin()]

    cols_current = st.columns(4)
    for i, risk in enumerate([r.replace("_risk_prob", "") for r in df_final.columns if r.endswith("_risk_prob")]):
        prob = current_row[risk + "_risk_prob"]
        alert = current_row[risk + "_risk_alert"]
        cols_current[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{prob:.2f}",
            delta=alert
        )

    # -------------------------------
    # Vertical space before Recommended Actions
    # -------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------------------------------
    # Recommended Actions / Preparations
    # -------------------------------
    st.subheader("🛠 Recommended Actions / Preparations")
    risk_levels = ["Low", "Moderate", "Severe"]

    for risk in [r.replace("_risk_prob", "") for r in df_final.columns if r.endswith("_risk_prob")]:
        prob = current_row[risk + "_risk_prob"]
        if prob < 0.3:
            level = "Low"
            actions = f"- Monitor local weather updates\n- Stay alert but no immediate action required for {risk}"
        elif prob < 0.7:
            level = "Moderate"
            actions = f"- Prepare emergency kits\n- Check evacuation routes\n- Stay updated for {risk}"
        else:
            level = "Severe"
            actions = f"- Evacuate if necessary\n- Follow local authority instructions\n- Secure property for {risk}"

        st.markdown(f"**{risk.replace('_', ' ').title()} Risk:** {level}")
        st.markdown(actions)
        st.markdown("<br>", unsafe_allow_html=True)  # space between each risk

    # -------------------------------
    # Vertical space before Detailed Output
    # -------------------------------
    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------
    # Detailed Output Table
    # -------------------------------
    st.subheader("🧾 Detailed Output (Forecast)")
    feature_cols = [c for c in df_final.columns if "risk_prob" in c or "risk_alert" in c or c == "date"]
    st.dataframe(df_final[feature_cols], use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
