"""
Anomaly detection with Isolation Forest over the AI4I sensor space.

Identifies abnormal operating behaviour, produces an anomaly score per row
(outlierness in [0, 1], higher = more anomalous), persists the fitted model,
the per-row results CSV and a visualisation.
"""
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)
ANOMALY_FEATURES = config.RAW_NUMERIC + ["temp_difference", "power_watts"]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "temp_difference" not in df:
        df["temp_difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    if "power_watts" not in df:
        df["power_watts"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60.0
    return df[ANOMALY_FEATURES]


def train_anomaly_detector(df: pd.DataFrame, contamination: float = 0.034):
    """Fit scaler + IsolationForest. contamination ~ observed failure rate (3.4%)."""
    X = _features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=config.RANDOM_STATE, n_jobs=-1)
    iso.fit(X_scaled)
    bundle = {"scaler": scaler, "model": iso, "features": ANOMALY_FEATURES,
              "contamination": contamination}
    joblib.dump(bundle, os.path.join(config.MODELS_DIR, "anomaly_model.pkl"))
    logger.info("Anomaly detector saved (contamination=%.4f)", contamination)
    return bundle


def detect_anomalies(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    if bundle is None:
        bundle = joblib.load(os.path.join(config.MODELS_DIR, "anomaly_model.pkl"))
    X = _features(df)
    X_scaled = bundle["scaler"].transform(X)
    raw_score = bundle["model"].decision_function(X_scaled)   # higher = more normal
    pred = bundle["model"].predict(X_scaled)                  # -1 anomaly, 1 normal
    # Normalise to [0, 1] with 1 = most anomalous
    norm = (raw_score.max() - raw_score) / (raw_score.max() - raw_score.min() + 1e-12)
    out = df.copy().reset_index(drop=True)
    out["temp_difference"] = X["temp_difference"].values
    out["power_watts"] = X["power_watts"].values
    out["anomaly_score"] = norm.round(6)
    out["is_anomaly"] = (pred == -1).astype(int)
    return out


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    bundle = train_anomaly_detector(df)
    results = detect_anomalies(df, bundle)
    keep_cols = ([c for c in ["UDI", "Type"] if c in results.columns]
                 + ANOMALY_FEATURES + ["Machine failure", "anomaly_score", "is_anomaly"])
    results[keep_cols].to_csv(os.path.join(config.PREDICTIONS_DIR, "anomaly_results.csv"), index=False)
    n_anom = int(results["is_anomaly"].sum())
    overlap = int(((results["is_anomaly"] == 1) & (results["Machine failure"] == 1)).sum())
    logger.info("Anomalies detected: %d (%.2f%%); overlapping with actual failures: %d",
                n_anom, 100 * n_anom / len(results), overlap)
    _plot(results)
    return results


def _plot(results: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    normal = results["is_anomaly"] == 0
    axes[0].scatter(results.loc[normal, "Rotational speed [rpm]"],
                    results.loc[normal, "Torque [Nm]"], s=8, alpha=0.35,
                    c="#2e86ab", label=f"Normal ({normal.sum()})")
    axes[0].scatter(results.loc[~normal, "Rotational speed [rpm]"],
                    results.loc[~normal, "Torque [Nm]"], s=14, alpha=0.8,
                    c="#d64541", label=f"Anomaly ({(~normal).sum()})")
    axes[0].set_xlabel("Rotational speed [rpm]"); axes[0].set_ylabel("Torque [Nm]")
    axes[0].set_title("Anomalies in RPM–Torque Space"); axes[0].legend()

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    pts = pca.fit_transform(StandardScaler().fit_transform(_features(results)))
    axes[1].scatter(pts[normal, 0], pts[normal, 1], s=8, alpha=0.35, c="#2e86ab", label="Normal")
    axes[1].scatter(pts[~normal, 0], pts[~normal, 1], s=14, alpha=0.8, c="#d64541", label="Anomaly")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].set_title("Anomalies in PCA Space"); axes[1].legend()

    sns.histplot(results["anomaly_score"], bins=50, ax=axes[2], color="#6d6875")
    axes[2].set_xlabel("Anomaly score (1 = most anomalous)")
    axes[2].set_title("Anomaly Score Distribution")
    fig.suptitle("Isolation Forest Anomaly Detection", fontweight="bold")
    fig.savefig(os.path.join(config.VIZ_DIR, "anomaly_detection.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
