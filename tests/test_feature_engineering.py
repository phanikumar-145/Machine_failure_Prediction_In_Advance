import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import feature_engineering as fe


def _sample_df():
    return pd.DataFrame({
        "UDI": [1, 2, 3, 4, 5, 6],
        "Product ID": ["a", "b", "c", "d", "e", "f"],
        "Type": ["L", "M", "H", "L", "M", "L"],
        "Air temperature [K]": [298.0, 300.0, 299.0, 301.0, 298.5, 302.0],
        "Process temperature [K]": [308.0, 310.0, 309.0, 311.0, 309.5, 312.0],
        "Rotational speed [rpm]": [1500, 1600, 1400, 1550, 1700, 1450],
        "Torque [Nm]": [40.0, 45.0, 35.0, 50.0, 38.0, 42.0],
        "Tool wear [min]": [0, 10, 20, 30, 40, 50],
        "Machine failure": [0, 0, 0, 1, 0, 0],
    })


def test_physics_features_math():
    df = fe.add_physics_features(_sample_df())
    assert df.loc[0, "temp_difference"] == pytest.approx(10.0)
    import numpy as np
    expected_power = 40.0 * 1500 * 2 * np.pi / 60.0
    assert df.loc[0, "power_watts"] == pytest.approx(expected_power)
    assert df.loc[0, "tool_wear_torque"] == pytest.approx(0.0)
    assert "torque_per_rpm" in df.columns


def test_lag_features_are_past_only():
    df = fe.add_sequence_features(_sample_df())
    # Row i lag1 must equal row i-1 raw value (strictly past information)
    for i in range(1, len(df)):
        assert df.loc[i, "Torque [Nm]_lag1"] == pytest.approx(df.loc[i - 1, "Torque [Nm]"])


def test_rolling_features_no_future_leakage():
    df = fe.add_sequence_features(_sample_df(), window=3)
    # Causal rolling mean over current + past only
    expected = df["Torque [Nm]"].iloc[:2].mean()
    assert df.loc[1, "Torque [Nm]_roll_mean_3"] == pytest.approx(expected)


def test_engineered_output_persists(tmp_path):
    out = tmp_path / "eng.csv"
    df = fe.engineer_features(_sample_df(), save_path=str(out))
    assert out.exists()
    for col in ["temp_difference", "power_watts", "Type_L", "Type_M", "Type_H", "tool_wear_stage"]:
        assert col in df.columns


def test_model_matrix_order_and_columns():
    df = fe.engineer_features(_sample_df(), save_path="/tmp/_eng_test.csv")
    X = fe.make_model_matrix(df)
    assert list(X.columns) == [config.clean_feature_name(c) for c in config.MODEL_FEATURES]
    assert len(X) == len(df)
