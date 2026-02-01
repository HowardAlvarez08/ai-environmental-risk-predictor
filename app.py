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
        df_raw = fetch_real_time_weather(latitude, longitude, forecast_days)

    st.success("✅ Weather data fetched")

    with st.spinner("Engineering features..."):
        df_features = engineer_features(df_raw)

    with st.spinner("Predicting risks..."):
        df_pred = predict_risks(df_features, models)

    with st.spinner("Applying recommendations..."):
        df_final = apply_risk_alerts(df_pred)

    # -------------------------------
    # Current Hour Dashboard
    # -------------------------------
    now_ph = datetime.now(pytz.timezone("Asia/Manila"))
    now_ph_str = now_ph.strftime("%I:%M %p")
    st.subheader(f"⏱ Current Hour Risk Status ({now_ph_str})")

    # Ensure date column is timezone-aware for comparison
    if df_final["date"].dt.tz is None:
        df_final["date"] = df_final["date"].dt.tz_localize("Asia/Manila")

    # Find the row closest to current hour
    df_final['hour_diff'] = abs(df_final['date'] - now_ph)
    current_row = df_final.loc[df_final['hour_diff'].idxmin()]

    # List of risk columns
    risk_columns = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]

    # Display current risk metrics
    cols_current = st.columns(len(risk_columns))
    for i, risk in enumerate(risk_columns):
        prob = current_row[risk + "_risk_prob"]
        alert = current_row.get(risk + "_risk_alert", "")
        label = risk.replace("_", " ").title()

        # Color-coded based on probability
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
    st.subheader("🛠 Recommended Actions / Preparations")
    for risk in risk_columns:
        prob = current_row[risk + "_risk_prob"]
        label = risk.replace("_", " ").title()

        if prob < 0.3:
            action_text = (
                f"✅ **{label} – Low Risk:** Stay aware and monitor updates. "
                "No immediate action required."
            )
        elif prob < 0.7:
            action_text = (
                f"⚠️ **{label} – Moderate Risk:** Prepare emergency kits, "
                "check your surroundings, and avoid risky areas. Stay alert."
            )
        else:
            action_text = (
                f"🚨 **{label} – Severe Risk:** Take immediate precautions. "
                "Follow local government advisories, move to safe areas if necessary, "
                "stock essential supplies, and avoid travel."
            )

        st.markdown(action_text)

    # -------------------------------
    # Detailed Output
    # -------------------------------
    st.subheader("🧾 Detailed Output (Forecast)")

    # Remove duplicates
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # Keep only relevant columns
    feature_cols = [c for c in df_final.columns if "risk_prob" in c or "risk_alert" in c]
    display_cols = ["date"] + feature_cols
    st.dataframe(df_final[display_cols].tail(24), use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
