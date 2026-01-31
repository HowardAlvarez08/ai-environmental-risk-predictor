# app.py
import streamlit as st
import pandas as pd
from src.data_fetch import fetch_real_time_weather
from src.feature_engineering import engineer_features
from src.models_loader import load_models
from src.predict_risks import predict_risks
from src.rule_recommendations import apply_alerts
import pickle

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(
    page_title="AI Environmental Risk Predictor",
    layout="wide"
)
st.title("🌦️ AI Environmental Risk Predictor")
st.markdown("Predict real-time flood, rain, storm, and landslide risks.")

# ------------------------------
# Sidebar options
# ------------------------------
st.sidebar.header("Settings")
lat = st.sidebar.number_input("Latitude", value=14.5995)
lon = st.sidebar.number_input("Longitude", value=120.9842)
timezone = st.sidebar.selectbox("Timezone", ["Asia/Manila", "Asia/Singapore"], index=0)

# ------------------------------
# Step 1: Fetch data
# ------------------------------
st.info("Fetching real-time weather data...")
df_weather = fetch_real_time_weather(lat=lat, lon=lon, timezone=timezone)
st.success("✅ Weather data fetched!")

st.dataframe(df_weather.head(5))

# ------------------------------
# Step 2: Feature engineering
# ------------------------------
st.info("Engineering features...")
df_features = engineer_features(df_weather)
st.success("✅ Features ready")

st.dataframe(df_features.head(5))

# ------------------------------
# Step 3: Load models and scaler
# ------------------------------
st.info("Loading trained models...")

models = load_models("models")

# Load scaler and feature lists (assume saved with pickle)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/continuous_features.pkl", "rb") as f:
    continuous_features = pickle.load(f)

with open("models/binary_features.pkl", "rb") as f:
    binary_features = pickle.load(f)

with open("models/cyclical_features.pkl", "rb") as f:
    cyclical_features = pickle.load(f)

st.success("✅ Models loaded!")

# ------------------------------
# Step 4: Predict risks
# ------------------------------
st.info("Predicting risks...")
df_predicted = predict_risks(
    df_features,
    models,
    scaler,
    continuous_features,
    binary_features=binary_features,
    cyclical_features=cyclical_features
)
st.success("✅ Predictions done!")

# ------------------------------
# Step 5: Apply alerts
# ------------------------------
st.info("Generating rule-based alerts...")
df_alerts = apply_alerts(df_predicted)
st.success("✅ Alerts ready!")

# ------------------------------
# Step 6: Display results
# ------------------------------
st.header("Risk Predictions & Alerts")
alert_cols = [col for col in df_alerts.columns if "_alert" in col]
st.dataframe(df_alerts[alert_cols].tail(24))  # last 24 hours

# Optional: download results
st.download_button(
    label="📥 Download CSV",
    data=df_alerts.to_csv(index=False),
    file_name="real_time_risks.csv",
    mime="text/csv"
)
