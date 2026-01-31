# -------------------------------
# Display Results
# -------------------------------
st.subheader("📊 Latest Risk Assessment")

latest = df_final.iloc[-1]

cols = st.columns(4)

for i, risk in enumerate(
    [c.replace("_risk_prob", "") for c in df_final.columns if c.endswith("_risk_prob")]
):
    alert_col = risk + "_risk_alert"
    # ✅ Use .get() to avoid KeyError if column missing
    cols[i].metric(
        label=risk.replace("_", " ").title(),
        value=f"{latest[risk + '_risk_prob']:.2f}",
        delta=latest.get(alert_col, 0)  # default to 0 if alert column missing
    )

# -------------------------------
# Detailed Output (Latest Date)
# -------------------------------
st.subheader("🧾 Detailed Output (Latest Available)")

# Get the latest timestamp in your data
latest_date = df_final['date'].max()
df_latest = df_final[df_final['date'] == latest_date]

# Optional: select columns to show, fewer than all
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
