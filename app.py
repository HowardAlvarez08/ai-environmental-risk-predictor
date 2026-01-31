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
    # Display Results
    # -------------------------------
    st.subheader("📊 Latest Risk Assessment")

    latest = df_final.iloc[-1]

    # Get all risk names dynamically
    risk_names = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]

    # Create columns dynamically
    cols = st.columns(len(risk_names))

    for i, risk in enumerate(risk_names):
        risk_prob_col = risk + "_risk_prob"
        alert_col = risk + "_alert"

        # Safely handle missing alert column
        alert_value = latest[alert_col] if alert_col in df_final.columns else None

        cols[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{latest[risk_prob_col]:.2f}" if risk_prob_col in df_final else "N/A",
            delta=alert_value
        )

    st.subheader("🧾 Detailed Output")
    st.dataframe(df_final.tail(24))

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
