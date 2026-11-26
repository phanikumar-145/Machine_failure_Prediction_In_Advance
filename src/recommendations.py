"""
Rule-based maintenance recommendations driven by real model outputs and
sensor readings. Used by the dashboard and available for batch reporting.
"""
import sys, os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_recommendations(reading: dict, failure_probability: float,
                             anomaly_score: float | None = None) -> list:
    """
    reading: dict with the raw sensor fields (validated upstream).
    Returns a prioritised list of actionable recommendation strings.
    """
    recs = []
    air = float(reading["Air temperature [K]"])
    proc = float(reading["Process temperature [K]"])
    rpm = float(reading["Rotational speed [rpm]"])
    torque = float(reading["Torque [Nm]"])
    wear = float(reading["Tool wear [min]"])
    temp_diff = proc - air
    power_kw = torque * rpm * 2 * 3.141592653589793 / 60.0 / 1000.0

    # --- model-risk driven actions ---------------------------------------
    if failure_probability >= config.RISK_HIGH:
        recs.append("CRITICAL: failure probability >= 80% - stop the machine at the next safe opportunity and perform a full inspection before further operation.")
    elif failure_probability >= config.RISK_MEDIUM:
        recs.append("HIGH: failure probability >= 60% - schedule maintenance within 24-48 hours; prepare spare tooling and inspect the spindle/tool assembly.")
    elif failure_probability >= config.RISK_LOW:
        recs.append("WATCH: elevated failure probability - increase inspection frequency and monitor torque/temperature closely over the next shift.")
    else:
        recs.append("Operating within normal risk envelope - continue routine monitoring.")

    # --- anomaly-driven actions ------------------------------------------
    if anomaly_score is not None:
        if anomaly_score >= 0.70:
            recs.append("Anomaly detector flags this operating point as strongly abnormal - verify sensor calibration and inspect for mechanical looseness/misalignment.")
        elif anomaly_score >= 0.45:
            recs.append("Moderately anomalous sensor pattern - compare against the unit's recent baseline and check for process drift.")

    # --- physics / domain rules -------------------------------------------
    if wear >= 200:
        recs.append(f"Tool wear at {wear:.0f} min (>200) - schedule immediate tool replacement; wear strongly correlates with tool-wear failure (TWF).")
    elif wear >= 150:
        recs.append(f"Tool wear at {wear:.0f} min - plan a tool change soon; inspect cutting edge for chipping.")

    if temp_diff < 9.5:
        recs.append(f"Temperature difference only {temp_diff:.1f} K (<9.5 K) - poor heat dissipation; check coolant flow, fans and heat exchangers (HDF risk).")
    elif temp_diff > 12.5:
        recs.append(f"Large process-air temperature difference {temp_diff:.1f} K - verify thermal insulation and check for overheating process conditions.")

    if torque > 60:
        recs.append(f"Torque at {torque:.1f} Nm (>60) - risk of overstrain failure (OSF); reduce feed rate/load and inspect driveline for binding.")
    if rpm < 1350 and torque > 50:
        recs.append("Low speed with high torque - power failure (PWF) regime; check motor current draw and power transmission.")
    if power_kw > 9.0:
        recs.append(f"Mechanical power ~{power_kw:.1f} kW is elevated - verify motor loading and check for excessive resistance.")
    if rpm > 2700:
        recs.append(f"Rotational speed {rpm:.0f} rpm is extreme - check balancing, bearings and resonance before continued high-speed operation.")

    if len(recs) == 1:
        recs.append("No abnormal sensor conditions detected - log this reading and continue scheduled preventive maintenance.")
    return recs


def batch_recommendations(df, probability_col: str = "failure_probability",
                          anomaly_col: str | None = None) -> pd.Series:
    """Apply generate_recommendations row-wise; returns a Series of joined strings."""
    import pandas as pd
    recs = []
    for _, row in df.iterrows():
        reading = row.to_dict()
        prob = float(row.get(probability_col, 0.0))
        score = float(row[anomaly_col]) if anomaly_col and anomaly_col in df.columns else None
        recs.append(" | ".join(generate_recommendations(reading, prob, score)))
    return pd.Series(recs, index=df.index, name="recommendations")
