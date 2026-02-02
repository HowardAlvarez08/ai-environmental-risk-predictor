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

st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=7, value=1)
hour_options = {
    "Now": 0,
    "+1 hour": 1,
    "+2 hours": 2,
    "+4 hours": 4,
    "+8 hours": 8,
    "+12 hours": 12,
    "+16 hours": 16,
    "+24 hours": 24,
}
selected_offset_label = st.sidebar.selectbox("Select Forecast Time", list(hour_options.keys()))
offset_hours = hour_options[selected_offset_label]

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
    # Forecast Hour Dashboard
    # -------------------------------
    st.subheader("⏱ Forecast Hour Risk Status")

    # Target datetime
    now_ph = pd.Timestamp.now(tz="Asia/Manila")
    target_time = now_ph + pd.Timedelta(hours=offset_hours)
    target_time_str = target_time.strftime("%I:%M %p")
    st.write(f"Showing risk status for: **{target_time_str}**")

    # Ensure date column is timezone-aware
    if df_final["date"].dt.tz is None:
        df_final["date"] = df_final["date"].dt.tz_localize("Asia/Manila")

    # Find the row closest to target_time
    df_final['hour_diff'] = abs(df_final['date'] - target_time)
    forecast_row = df_final.loc[df_final['hour_diff'].idxmin()]

    # List of risk columns
    risk_columns = [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]

    # Display risk metrics (color-coded)
    cols_forecast = st.columns(len(risk_columns))
    for i, risk in enumerate(risk_columns):
        prob = forecast_row[risk + "_risk_prob"]
        alert = forecast_row.get(risk + "_risk_alert", "")
        label = risk.replace("_", " ").title()

        if prob < 0.3:
            color = "✅"
        elif prob < 0.7:
            color = "⚠️"
        else:
            color = "🚨"

        cols_forecast[i].metric(
            label=label,
            value=f"{prob:.2f}",
            delta=f"{color} {alert}"
        )

    # -------------------------------
    # Recommended Actions / Preparations
    # -------------------------------
    st.subheader("🛠 Recommended Actions / Preparations")
    for risk in risk_columns:
        prob = forecast_row[risk + "_risk_prob"]
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
    # Detailed Output Table
    # -------------------------------
    st.subheader("🧾 Detailed Forecast Data (Relevant Columns)")

    feature_cols = [
        c for c in df_final.columns 
        if c.endswith("_risk_prob") or c.endswith("_risk_alert") or c == "date"
    ]
    df_final_unique = df_final.loc[:, ~df_final.columns.duplicated()]
    display_cols = [c for c in feature_cols if c in df_final_unique.columns]
    st.dataframe(df_final_unique[display_cols].tail(24), use_container_width=True)

else:
    st.info("👈 Click **Fetch & Predict** to run the model.")
