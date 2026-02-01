# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
    # Ensure timezone-aware datetime
    # -------------------------------
    if df_final["date"].dt.tz is None:
        df_final["date"] = df_final["date"].dt.tz_localize("Asia/Manila")
    else:
        df_final["date"] = df_final["date"].dt.tz_convert("Asia/Manila")

    # -------------------------------
    # Current Hour Risk Dashboard
    # -------------------------------
    st.subheader("⏰ Current Hour Risk Status")

    # Dropdown for offset selection
    offset_options = {
        "Now": 0,
        "+1 hour": 1,
        "+2 hours": 2,
        "+4 hours": 4,
        "+8 hours": 8,
        "+12 hours": 12,
        "+16 hours": 16,
        "+24 hours": 24
    }
    selected_offset_label = st.selectbox("Select time offset from now", list(offset_options.keys()))
    offset_hours = offset_options[selected_offset_label]

    # Current PH time
    now_ph = pd.Timestamp.now(tz="Asia/Manila")
    target_time = now_ph + timedelta(hours=offset_hours)
    target_time = target_time.replace(minute=0, second=0, microsecond=0)
    now_ph_str = target_time.strftime("%I:%M %p")
    st.write(f"Selected Time: {now_ph_str}")

    # Find closest row to selected hour
    df_final['hour_diff'] = abs(df_final['date'] - target_time)
    if not df_final.empty:
        current_row = df_final.loc[df_final['hour_diff'].idxmin()]
    else:
        st.warning("No data available for the selected time.")
        current_row = None

    # Show risk metrics if row found
    if current_row is not None:
        cols_current = st.columns(4)
        for i, risk in enumerate(["flood", "rain", "storm", "landslide"]):
            prob = current_row.get(f"{risk}_risk_prob", np.nan)
            alert = current_row.get(f"{risk}_risk_alert", "N/A")
            cols_current[i].metric(
                label=risk.replace("_", " ").title(),
                value=f"{prob:.2f}",
                delta=alert
            )

        # -------------------------------
        # Recommended Actions
        # -------------------------------
        st.subheader("📝 Recommended Actions / Preparations")

        def risk_actions(risk_prob):
            if risk_prob < 0.3:
                return "Low risk: Normal precautions. Stay updated."
            elif risk_prob < 0.7:
                return "Moderate risk: Prepare emergency kits, monitor weather updates."
            else:
                return "Severe risk: Follow official evacuation orders, avoid travel, secure property."

        for risk in ["flood", "rain", "storm", "landslide"]:
            prob = current_row.get(f"{risk}_risk_prob", np.nan)
            st.markdown(f"**{risk.title()}**: {risk_actions(prob)}")

    # -------------------------------
    # Detailed Output
    # -------------------------------
    st.subheader("🧾 Detailed Hourly Forecast Data")
    feature_cols = [c for c in df_final.columns if "risk_prob" in c or "risk_alert" in c]
    display_cols = ["date"] + feature_cols
    # remove duplicates if any
    display_cols = list(dict.fromkeys(display_cols))
    st.dataframe(df_final[display_cols].sort_values("date"), use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
