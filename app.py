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
    # Fetch weather
    with st.spinner("Fetching real-time weather data..."):
        df_raw = fetch_real_time_weather(latitude, longitude)
    st.success("✅ Weather data fetched")

    # Feature engineering
    with st.spinner("Engineering features..."):
        df_features = engineer_features(df_raw)

    # Risk prediction
    with st.spinner("Predicting risks..."):
        df_pred = predict_risks(df_features, models)

    # Apply alerts
    with st.spinner("Applying recommendations..."):
        df_final = apply_risk_alerts(df_pred)

    # -------------------------------
    # Display Latest Risk Assessment
    # -------------------------------
    st.subheader("📊 Latest Risk Assessment")

    latest = df_final.iloc[-1]

    # Only pick columns that end with '_risk_prob'
    risk_cols = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]
    cols = st.columns(len(risk_cols))

    for i, risk in enumerate(risk_cols):
        alert_col = risk + "_alert"
        # Use 0 if alert column is missing
        delta_val = latest[alert_col] if alert_col in df_final.columns else 0
        cols[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{latest[risk + '_risk_prob']:.2f}",
            delta=delta_val
        )

    # -------------------------------
    # Latest Day Detailed Output (24h)
    # -------------------------------
    st.subheader("🧾 Detailed Output (Latest Day)")

    # Convert datetime column safely
    if 'datetime' in df_final.columns:
        df_final['date_only'] = pd.to_datetime(df_final['datetime']).dt.date
    elif 'time' in df_final.columns:
        df_final['date_only'] = pd.to_datetime(df_final['time']).dt.date
    else:
        st.error("No datetime column found in data!")
        st.stop()

    latest_day = df_final['date_only'].max()
    df_today = df_final[df_final['date_only'] == latest_day]

    # Select only relevant columns to avoid a huge table
    selected_cols = [
        'date_only', 'temperature_mean', 'relative_humidity_mean', 
        'wind_speed_mean', 'precipitation_sum',
        'flood_risk_prob', 'flood_risk_alert',
        'rain_risk_prob', 'rain_risk_alert',
        'storm_risk_prob', 'storm_risk_alert',
        'landslide_risk_prob', 'landslide_risk_alert',
        'overall_alert'
    ]
    # Keep only existing columns (some might be missing)
    selected_cols = [c for c in selected_cols if c in df_today.columns]

    df_today = df_today[selected_cols]

    st.dataframe(df_today)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
