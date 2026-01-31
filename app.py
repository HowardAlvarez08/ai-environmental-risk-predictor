# app.py

import streamlit as st
import pandas as pd
import numpy as np

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
        df_raw = fetch_real_time_weather(latitude, longitude)

    st.success("✅ Weather data fetched")

    with st.spinner("Engineering features..."):
        df_features = engineer_features(df_raw)

    with st.spinner("Predicting risks..."):
        df_pred = predict_risks(df_features, models)

    with st.spinner("Applying recommendations..."):
        df_final = apply_risk_alerts(df_pred)

    # -------------------------------
    # Latest Day Selection (24-hour view)
    # -------------------------------
    df_final['date_only'] = pd.to_datetime(df_final['datetime']).dt.date
    latest_day = df_final['date_only'].max()
    df_today = df_final[df_final['date_only'] == latest_day]

    st.subheader(f"📊 Latest Risk Assessment ({latest_day})")

    # Show max risk of the day as summary
    risk_cols = [c.replace("_risk_prob", "") for c in df_today.columns if c.endswith("_risk_prob")]
    cols = st.columns(len(risk_cols))

    latest = df_today.iloc[-1]  # Or use max aggregation per risk
    for i, risk in enumerate(risk_cols):
        # Use max risk probability for summary
        max_prob = df_today[risk + "_risk_prob"].max()
        # Use max alert for delta (or last alert)
        max_alert = df_today[risk + "_risk_alert"].max() if (risk + "_risk_alert") in df_today.columns else None

        cols[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{max_prob:.2f}",
            delta=max_alert
        )

    # -------------------------------
    # Detailed 24-hour table
    # -------------------------------
    st.subheader("🧾 Detailed Output (Latest Day)")
    selected_cols = [
        'datetime',  # show timestamp
        # keep only main numeric variables if too many columns exist
    ]

    # Add risk probs and alerts
    for risk in risk_cols:
        selected_cols.append(risk + "_risk_prob")
        if risk + "_risk_alert" in df_today.columns:
            selected_cols.append(risk + "_risk_alert")

    # Include overall alert if exists
    if 'overall_alert' in df_today.columns:
        selected_cols.append('overall_alert')

    # Keep only existing columns to avoid KeyError
    selected_cols = [c for c in selected_cols if c in df_today.columns]
    df_today = df_today[selected_cols]

    st.dataframe(df_today.reset_index(drop=True))

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
