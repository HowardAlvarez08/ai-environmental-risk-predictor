# app.py

import streamlit as st
import pandas as pd
from datetime import datetime

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
    # Display Metrics
    # -------------------------------
    st.subheader("📊 Latest Risk Assessment")
    latest = df_final.iloc[-1]

    # Identify risk names dynamically
    risk_names = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]
    cols = st.columns(len(risk_names))

    for i, risk in enumerate(risk_names):
        prob_col = f"{risk}_risk_prob"
        alert_col = f"{risk}_risk_alert"
        prob_value = latest[prob_col] if prob_col in latest else None
        alert_value = latest[alert_col] if alert_col in latest else None

        # Show metric safely
        cols[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{prob_value:.2f}" if prob_value is not None else "N/A",
            delta=alert_value if alert_value is not None else "N/A"
        )

    # -------------------------------
    # Detailed Output
    # -------------------------------
    st.subheader("🧾 Detailed Output (Today)")

    # Ensure 'date' column is datetime
    if 'date' in df_final.columns:
        df_final['date'] = pd.to_datetime(df_final['date'])
        today = pd.Timestamp(datetime.now().date())
        df_today = df_final[df_final['date'].dt.date == today.date()]
    else:
        df_today = df_final.copy()  # fallback

    # Desired columns (only keep those existing)
    desired_cols = [
        'date', 'temperature_mean', 'relative_humidity_mean', 'wind_speed_mean',
        'flood_risk_prob', 'flood_risk_alert',
        'rain_risk_prob', 'rain_risk_alert',
        'storm_risk_prob', 'storm_risk_alert',
        'landslide_risk_prob', 'landslide_risk_alert',
        'overall_alert'
    ]
    existing_cols = [col for col in desired_cols if col in df_today.columns]
    df_today = df_today[existing_cols]

    st.dataframe(df_today)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
