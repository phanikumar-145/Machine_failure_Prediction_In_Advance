"""
Feature engineering for predictive maintenance.

Instantaneous (physics-inspired) features - safe for single-row inference:
  * temp_difference   : process - air temperature (heat-dissipation margin)
  * power_watts       : P = torque * angular velocity
  * tool_wear_torque  : wear x torque (overstrain proxy)
  * torque_per_rpm    : mechanical stress per revolution

Chronological features - computed on UDI order, strictly past-only
(shift-based lags and causal rolling windows, no look-ahead):
  * <sensor>_lag1, <sensor>_diff, <sensor>_roll_mean_5, <sensor>_roll_std_5

AI4I 2020 has NO timestamps; it is a snapshot dataset. The rolling/lag features
are therefore computed over the file (UDI) order as a stand-in sequence and are
used only for analysis outputs, NOT for the classification model - training the
model on them would make single-row dashboard predictions inconsistent. The
genuine chronological degradation analysis lives in src/forecasting.py (C-MAPSS).
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

SENSOR_FOR_SEQUENCE = config.SENSOR_COLUMNS


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    angular_velocity = df["Rotational speed [rpm]"] * 2 * np.pi / 60.0
    df["power_watts"] = df["Torque [Nm]"] * angular_velocity
    df["tool_wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["torque_per_rpm"] = df["Torque [Nm]"] / df["Rotational speed [rpm]"].clip(lower=1)
    return df


def add_sequence_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Past-only lag / rolling features, ordered by UDI. No future information used."""
    df = df.sort_values("UDI").reset_index(drop=True).copy()
    for col in SENSOR_FOR_SEQUENCE:
        prev = df[col].shift(1)
        # First row has no history -> fall back to the current reading (documented).
        df[f"{col}_lag1"] = prev.fillna(df[col])
        df[f"{col}_diff"] = df[col].diff().fillna(0.0)
        df[f"{col}_roll_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std_{window}"] = df[col].rolling(window, min_periods=1).std().fillna(0.0)
    return df


def add_wear_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Tool-wear progression indicator: normalised wear and wear stage bucket."""
    df = df.copy()
    max_wear = df["Tool wear [min]"].max()
    df["tool_wear_pct"] = df["Tool wear [min]"] / max_wear if max_wear else 0.0
    # 0 = fresh, 1 = early, 2 = mid, 3 = near end-of-life (>200 min)
    df["tool_wear_stage"] = pd.cut(
        df["Tool wear [min]"], bins=[-1, 50, 150, 200, np.inf],
        labels=[0, 1, 2, 3]).astype(int)
    return df


def engineer_features(df: pd.DataFrame, save_path: str = config.ENGINEERED_FEATURES) -> pd.DataFrame:
    """Full feature-engineering stage. Persists engineered_features.csv."""
    from src.preprocessing import encode_type  # local import to avoid cycles
    df = add_physics_features(df)
    df = add_wear_progression(df)
    df = add_sequence_features(df)
    df = encode_type(df)  # adds Type_H / Type_L / Type_M, drops Type
    df.to_csv(save_path, index=False)
    logger.info("Engineered features saved to %s (%d rows, %d cols)", save_path, df.shape[0], df.shape[1])
    return df


def make_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select & order exactly the columns the models were trained on."""
    missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Engineered dataframe is missing model feature(s): {missing}")
    matrix = df[config.MODEL_FEATURES].rename(columns=config.clean_feature_name)
    return matrix
