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

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=1,
    max_value=7,
    value=1,
    help="Select how many days ahead you want forecasts for."
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
    # Show current hour snapshot
    # -------------------------------
    st.subheader("⏰ Current Hour Risk Status")

    now_ph = datetime.now()  # already PH time if data_fetch converts timezone

    # Find row closest to current time
    df_final['hour_diff'] = abs(df_final['date'] - now_ph)
    current_row = df_final.loc[df_final['hour_diff'].idxmin()]

    cols_current = st.columns(4)
    risk_columns = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]

    for i, risk in enumerate(risk_columns):
        cols_current[i].metric(
            label=risk.replace("_", " ").title(),
            value=f"{current_row[risk + '_risk_prob']:.2f}",
            delta=current_row.get(risk + "_alert", 0)
        )

    df_final.drop(columns=['hour_diff'], inplace=True)

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("📊 Latest Risk Assessment (All Forecast Hours)")
    
    # Only relevant columns: date + risk probs + alerts
    relevant_cols = ['date'] + [f"{r}_risk_prob" for r in risk_columns] + [f"{r}_alert" for r in risk_columns]
    st.dataframe(df_final[relevant_cols])

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
