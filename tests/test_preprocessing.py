import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import preprocessing


def test_handle_missing_values_median_and_mode():
    df = pd.DataFrame({
        "Torque [Nm]": [10.0, np.nan, 30.0, 40.0],
        "Type": ["L", None, "L", "M"],
    })
    out = preprocessing.handle_missing_values(df)
    assert out["Torque [Nm]"].isna().sum() == 0
    assert out.loc[1, "Torque [Nm]"] == pytest.approx(np.median([10.0, 30.0, 40.0]))
    assert out["Type"].isna().sum() == 0


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = preprocessing.remove_duplicates(df)
    assert len(out) == 2


def test_validate_dtypes_clips_and_types():
    df = pd.DataFrame({
        "Air temperature [K]": [298.1, 301.2],
        "Process temperature [K]": [308.6, 310.0],
        "Rotational speed [rpm]": [1551, 1408],
        "Torque [Nm]": [42.8, 46.3],
        "Tool wear [min]": [0, 3],
        "Machine failure": [0, 1],
        "TWF": [0, 0], "HDF": [0, 0], "PWF": [0, 0], "OSF": [0, 0], "RNF": [0, 0],
        "Type": ["l", "M "],
    })
    out = preprocessing.validate_dtypes(df)
    assert list(out["Type"]) == ["L", "M"]
    assert str(out["Machine failure"].dtype) == "int64"


def test_validate_dtypes_rejects_bad_type():
    df = pd.DataFrame({
        "Air temperature [K]": [298.1], "Process temperature [K]": [308.6],
        "Rotational speed [rpm]": [1551], "Torque [Nm]": [42.8],
        "Tool wear [min]": [0], "Machine failure": [0],
        "TWF": [0], "HDF": [0], "PWF": [0], "OSF": [0], "RNF": [0],
        "Type": ["X"],
    })
    with pytest.raises(ValueError):
        preprocessing.validate_dtypes(df)


def test_analyze_outliers_shape():
    df = pd.DataFrame({col: np.arange(100, dtype=float) + (1000.0 if col == "Torque [Nm]" else 0.0)
                       for col in config.RAW_NUMERIC})
    out = preprocessing.analyze_outliers(df)
    assert list(out["column"]) == config.RAW_NUMERIC
    assert (out["n_outliers"] >= 0).all()
    assert (out["outlier_pct"] >= 0).all()
