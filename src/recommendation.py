# src/recommendation.py

def risk_alert(prob: float) -> str:
    """
    Convert probability into human-readable alert level.
    """
    if prob < 0.1:
        return "Low Risk — No Action"
    elif prob < 0.3:
        return "Moderate Risk — Stay Alert"
    elif prob < 0.5:
        return "High Risk — Prepare Precautions"
    else:
        return "Severe Risk — Take Immediate Action"


def apply_risk_alerts(df):
    """
    Apply rule-based alerts to available risk probability columns.
    """
    df = df.copy()

    risk_prob_cols = [
        col for col in df.columns
        if col.endswith("_risk_prob")
    ]

    # Apply individual alerts
    for prob_col in risk_prob_cols:
        alert_col = prob_col.replace("_prob", "_alert")
        df[alert_col] = df[prob_col].apply(risk_alert)

    # Combined overall alert
    if risk_prob_cols:
        severity_order = [
            "Low Risk — No Action",
            "Moderate Risk — Stay Alert",
            "High Risk — Prepare Precautions",
            "Severe Risk — Take Immediate Action"
        ]

        alert_cols = [
            col.replace("_prob", "_alert")
            for col in risk_prob_cols
        ]

        def overall_alert(row):
            severities = [row[col] for col in alert_cols]
            highest = max(severities, key=lambda x: severity_order.index(x))
            return highest

        df["overall_alert"] = df.apply(overall_alert, axis=1)
    else:
        df["overall_alert"] = "No Risks Predicted"

    return df
