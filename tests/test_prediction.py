import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import prediction

BASE_RECORD = {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1538.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 108.0,
}


def test_validate_inputs_ok():
    out = prediction.validate_inputs(BASE_RECORD)
    assert out["Type"] == "M"
    assert out["Torque [Nm]"] == pytest.approx(40.0)


def test_validate_inputs_rejects_missing():
    bad = dict(BASE_RECORD)
    del bad["Torque [Nm]"]
    with pytest.raises(ValueError):
        prediction.validate_inputs(bad)


def test_validate_inputs_rejects_out_of_range():
    bad = dict(BASE_RECORD)
    bad["Torque [Nm]"] = 9999.0
    with pytest.raises(ValueError):
        prediction.validate_inputs(bad)


def test_validate_inputs_rejects_bad_type():
    bad = dict(BASE_RECORD)
    bad["Type"] = "Q"
    with pytest.raises(ValueError):
        prediction.validate_inputs(bad)


def test_build_feature_row_matches_training_order():
    row = prediction.build_feature_row(BASE_RECORD)
    assert len(row) == 1
    assert list(row.columns) == [config.clean_feature_name(c) for c in config.MODEL_FEATURES]


def test_predict_single_returns_valid_probability():
    result = prediction.predict_single(BASE_RECORD, model_name="best")
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["predicted_failure"] in (0, 1)
    assert result["risk_level"] in ("Low", "Medium", "High", "Critical")


def test_predict_single_high_risk_profile():
    """Extreme operating point should score at least as risky as the nominal one."""
    nominal = prediction.predict_single(BASE_RECORD, model_name="best")["failure_probability"]
    extreme = prediction.predict_single({**BASE_RECORD, "Tool wear [min]": 230.0,
                                         "Torque [Nm]": 70.0,
                                         "Rotational speed [rpm]": 1200.0},
                                        model_name="best")["failure_probability"]
    assert extreme >= nominal
