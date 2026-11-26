"""
Prediction interface used by the Streamlit dashboard and tests.

Loads the ACTUAL trained models + metadata and produces real predictions.
Feature order is taken from models/model_metadata.json (written during
training) so training and inference can never drift apart.
"""
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def load_metadata() -> dict:
    with open(config.MODEL_METADATA) as fh:
        return json.load(fh)


def load_model(name: str = "xgboost"):
    """name: 'xgboost' | 'random_forest' | 'logistic_regression' | 'best'"""
    meta = load_metadata()
    if name == "best":
        name = meta.get("best_model", "xgboost")
    files = {"xgboost": "xgboost_model.pkl", "random_forest": "random_forest.pkl",
             "logistic_regression": "logistic_regression.pkl"}
    if name not in files:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(files)} or 'best'.")
    return joblib.load(os.path.join(config.MODELS_DIR, files[name])), name


def load_anomaly_bundle() -> dict:
    return joblib.load(os.path.join(config.MODELS_DIR, "anomaly_model.pkl"))


def validate_inputs(record: dict) -> dict:
    """Coerce & validate a single raw machine reading. Raises ValueError on bad input."""
    required = {
        "Air temperature [K]": (250.0, 400.0),
        "Process temperature [K]": (250.0, 450.0),
        "Rotational speed [rpm]": (100.0, 10000.0),
        "Torque [Nm]": (0.0, 300.0),
        "Tool wear [min]": (0.0, 1000.0),
    }
    clean = {}
    for col, (lo, hi) in required.items():
        if col not in record:
            raise ValueError(f"Missing required field: '{col}'")
        try:
            value = float(record[col])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{col}' must be numeric, got {record[col]!r}")
        if not (lo <= value <= hi):
            raise ValueError(f"Field '{col}' = {value} outside plausible range [{lo}, {hi}]")
        clean[col] = value
    mtype = str(record.get("Type", "M")).strip().upper()
    if mtype not in {"L", "M", "H"}:
        raise ValueError(f"Type must be one of L/M/H, got {mtype!r}")
    clean["Type"] = mtype
    return clean


def build_feature_row(record: dict) -> pd.DataFrame:
    """Raw reading -> single-row model matrix with columns in training order."""
    clean = validate_inputs(record)
    row = {c: clean[c] for c in config.RAW_NUMERIC}
    row["temp_difference"] = row["Process temperature [K]"] - row["Air temperature [K]"]
    row["power_watts"] = row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2 * np.pi / 60.0
    row["tool_wear_torque"] = row["Tool wear [min]"] * row["Torque [Nm]"]
    row["torque_per_rpm"] = row["Torque [Nm]"] / max(row["Rotational speed [rpm]"], 1.0)
    row["Type_H"] = int(clean["Type"] == "H")
    row["Type_L"] = int(clean["Type"] == "L")
    row["Type_M"] = int(clean["Type"] == "M")
    meta = load_metadata()
    raw_features = meta.get("raw_feature_names", config.MODEL_FEATURES)
    matrix = pd.DataFrame([{f: row[f] for f in raw_features}])
    return matrix.rename(columns=config.clean_feature_name)


def risk_level(probability: float) -> str:
    if probability < config.RISK_LOW:
        return "Low"
    if probability < config.RISK_MEDIUM:
        return "Medium"
    if probability < config.RISK_HIGH:
        return "High"
    return "Critical"


def predict_single(record: dict, model_name: str = "best") -> dict:
    model, resolved = load_model(model_name)
    X = build_feature_row(record)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return {
        "model": resolved,
        "failure_probability": round(proba, 6),
        "predicted_failure": pred,
        "prediction_label": "FAILURE" if pred else "NORMAL",
        "risk_level": risk_level(proba),
        "machine_health": "Critical" if proba >= config.RISK_HIGH else
                          ("Warning" if proba >= config.RISK_MEDIUM else
                           ("Watch" if proba >= config.RISK_LOW else "Healthy")),
    }


def predict_dataframe(df: pd.DataFrame, model_name: str = "best") -> pd.DataFrame:
    """Batch prediction for a CSV of raw readings (AI4I schema). Adds prediction columns."""
    model, resolved = load_model(model_name)
    records = df.to_dict(orient="records")
    rows = [build_feature_row(r) for r in records]
    X = pd.concat(rows, ignore_index=True)
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["failure_probability"] = proba.round(6)
    out["predicted_failure"] = (proba >= 0.5).astype(int)
    out["risk_level"] = [risk_level(p) for p in proba]
    out["prediction_label"] = np.where(out["predicted_failure"] == 1, "FAILURE", "NORMAL")
    out["model_used"] = resolved
    return out
