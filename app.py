# app.py

import streamlit as st
import pandas as pd
import pytz
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
# Helper: color-coded alert
# -------------------------------
def format_alert(prob, alert_value):
    """
    Return a colored alert string based on risk probability or alert value
    """
    if alert_value == 0 or prob < 0.3:
        return "✅ Low"
    elif alert_value == 1 or prob < 0.7:
        return "⚠️ Moderate"
    else:
        return "❌ Severe"

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
    # Remove duplicate columns if any
    # -------------------------------
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # -------------------------------
    # Current hour risk status
    # -------------------------------
    st.subheader("⏰ Current Hour Risk Status")
    manila_tz = pytz.timezone("Asia/Manila")
    now_ph = datetime.now(manila_tz)
    now_ph_str = now_ph.strftime("%I:%M %p")
    st.write(f"Current Time: {now_ph_str}")

    # Ensure tz-aware
    df_final['date'] = pd.to_datetime(df_final['date']).dt.tz_localize(manila_tz, ambiguous='NaT', nonexistent='shift_forward')

    # Find the row closest to current hour
    df_final['hour_diff'] = abs(df_final['date'] - now_ph)
    current_row = df_final.loc[df_final['hour_diff'].idxmin()]

    # -------------------------------
    # Risk matrix display
    # -------------------------------
    risk_columns = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]

    st.markdown("<div style='display:flex; gap:2rem;'>", unsafe_allow_html=True)
    for risk in risk_columns:
        prob = current_row[risk + "_risk_prob"]
        alert_val = current_row.get(risk + "_alert", 0)
        alert_str = format_alert(prob, alert_val)
        
        # Metric-like vertical layout
        st.markdown(f"""
            <div style='text-align:center;'>
                <div style='font-size:24px; font-weight:bold;'>{prob:.2f}</div>
                <div style='font-size:18px;'>{alert_str}</div>
                <div style='margin-bottom:10px;'></div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    df_final.drop(columns=['hour_diff'], inplace=True)

    # -------------------------------
    # Detailed Output (relevant columns only)
    # -------------------------------
    feature_cols = [c for c in df_final.columns if "risk_prob" in c or "_alert" in c]
    display_cols = ["date"] + feature_cols
    st.subheader("🧾 Detailed Output")
    st.dataframe(df_final[display_cols], use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
