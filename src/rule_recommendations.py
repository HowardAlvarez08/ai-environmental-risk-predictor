# src/rule_recommendations.py
import pandas as pd

def risk_alert(prob):
    if prob < 0.1:
        return "Low Risk — No Action"
    elif prob < 0.3:
        return "Moderate Risk — Stay Alert"
    elif prob < 0.5:
        return "High Risk — Prepare Precautions"
    else:
        return "Severe Risk — Take Immediate Action"

def apply_alerts(df):
    risk_probs = [r for r in df.columns if "_risk_prob" in r]
    for r in risk_probs:
        df[r.replace("_prob", "_alert")] = df[r].apply(risk_alert)

    # Overall alert
    severity_order = ["Low Risk — No Action", "Moderate Risk — Stay Alert",
                      "High Risk — Prepare Precautions", "Severe Risk — Take Immediate Action"]
    def overall(row):
        if not risk_probs:
            return "No Risks Predicted"
        alerts = [row[r.replace("_prob","_alert")] for r in risk_probs]
        idx = max([severity_order.index(a) for a in alerts])
        return severity_order[idx]

    df['overall_alert'] = df.apply(overall, axis=1)
    return df
