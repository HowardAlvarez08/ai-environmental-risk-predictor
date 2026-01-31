# app.py

import streamlit as st
import pandas as pd

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
    # Display Latest Risk Assessment
    # -------------------------------
    st.subheader("📊 Latest Risk Assessment")

    latest = df_final.iloc[-1]

    cols = st.columns(4)

    for i, risk in enumerate(
        [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]
    ):
        alert_col = risk + "_risk_alert"
        # Use .get() to avoid KeyError if column missing
        cols[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{latest[risk + '_risk_prob']:.2f}",
            delta=latest.get(alert_col, 0)  # default to 0 if alert column missing
        )

    # -------------------------------
    # Detailed Output (Latest Available)
    # -------------------------------
    st.subheader("🧾 Detailed Output (Latest Available)")

    # Get the latest timestamp in your data
    latest_date = df_final['date'].max()
    df_latest = df_final[df_final['date'] == latest_date]

    # Select columns to show, fewer than all
    selected_cols = [
        'date', 'temperature_mean', 'relative_humidity_mean', 'precipitation_sum',
        'flood_risk_prob', 'flood_risk_alert',
        'rain_risk_prob', 'rain_risk_alert',
        'storm_risk_prob', 'storm_risk_alert',
        'landslide_risk_prob', 'landslide_risk_alert',
        'overall_alert'
    ]

    # Filter only available columns to prevent KeyError
    selected_cols = [c for c in selected_cols if c in df_latest.columns]

    df_latest = df_latest[selected_cols]

    st.dataframe(df_latest)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
