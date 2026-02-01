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

st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days:", min_value=1, max_value=7, value=1)

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
    # Time selection dropdown
    # -------------------------------
    now_ph = datetime.now(pytz.timezone("Asia/Manila"))
    time_options = {
        "Now": 0,
        "+1 hour": 1,
        "+2 hours": 2,
        "+4 hours": 4,
        "+8 hours": 8,
        "+12 hours": 12,
        "+16 hours": 16,
        "+24 hours": 24
    }
    selected_time_label = st.selectbox("Select time for risk assessment:", list(time_options.keys()))
    offset_hours = time_options[selected_time_label]
    target_time = now_ph + timedelta(hours=offset_hours)
    target_time_str = target_time.strftime("%I:%M %p")

    st.subheader(f"⏱ Risk Status at {target_time_str}")

    # Ensure 'date' column is timezone-aware
    if df_final["date"].dt.tz is None:
        df_final["date"] = df_final["date"].dt.tz_localize("Asia/Manila")

    # Find row closest to selected time
    df_final['hour_diff'] = abs(df_final['date'] - target_time)
    current_row = df_final.loc[df_final['hour_diff'].idxmin()]

    # -------------------------------
    # Current Hour Dashboard
    # -------------------------------
    risk_columns = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]
    cols_current = st.columns(len(risk_columns))

    for i, risk in enumerate(risk_columns):
        prob = current_row[risk + "_risk_prob"]
        alert = current_row.get(risk + "_risk_alert", "")
        label = risk.replace("_", " ").title()

        # Color-coded
        if prob < 0.3:
            color = "✅"
        elif prob < 0.7:
            color = "⚠️"
        else:
            color = "🚨"

        cols_current[i].metric(
            label=label,
            value=f"{prob:.2f}",
            delta=f"{color} {alert}"
        )

    # -------------------------------
    # Recommended Actions / Preparations
    # -------------------------------
    st.subheader("🛡 Recommended Actions / Preparations")

    for risk in risk_columns:
        prob = current_row[risk + "_risk_prob"]
        label = risk.replace("_", " ").title()

        if prob < 0.3:
            action = f"Low Risk: Stay informed, normal activities."
        elif prob < 0.7:
            action = f"Moderate Risk: Be cautious, prepare emergency kits and monitor weather updates."
        else:
            action = f"Severe Risk: Take immediate action, follow local authority instructions, avoid risky areas."

        st.markdown(f"**{label}:** {action}")

    # -------------------------------
    # Detailed Output Table
    # -------------------------------
    st.subheader("🧾 Detailed Output")
    feature_cols = [c for c in df_final.columns if "risk_prob" in c or "risk_alert" in c]
    display_cols = ["date"] + feature_cols
    st.dataframe(df_final[display_cols], use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
