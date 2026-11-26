"""
Dataset loading utilities.

Primary dataset : AI4I 2020 Predictive Maintenance Dataset (UCI, real data)
                  10,000 rows, 6 numeric features + product type, binary target.
Time-series data: NASA C-MAPSS FD001 turbofan engine degradation (real data).

Both datasets are used in *as published* form. If the raw files are missing,
`ensure_ai4i()` attempts a fresh download from the UCI repository. The C-MAPSS
files must be supplied in dataset/raw/ (see dataset/README.md); a clear error
is raised otherwise - no fabricated data is ever substituted silently.
"""
import io
import logging
import os
import urllib.request
import zipfile

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Column names for the C-MAPSS FD001 files (26 space-separated columns)
CMAPSS_COLUMNS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def ensure_ai4i(path: str = config.AI4I_RAW) -> str:
    """Return the path to the AI4I raw CSV, downloading it if missing."""
    if os.path.exists(path):
        return path
    logger.info("AI4I raw file not found at %s - attempting download from UCI ...", path)
    try:
        with urllib.request.urlopen(config.AI4I_URL, timeout=120) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    with zf.open(name) as src, open(path, "wb") as dst:
                        dst.write(src.read())
                    logger.info("Downloaded and extracted %s", name)
                    return path
        raise RuntimeError("Downloaded archive did not contain a CSV file.")
    except Exception as exc:  # pragma: no cover - depends on network
        raise FileNotFoundError(
            f"AI4I dataset missing at {path} and download failed ({exc}). "
            "See dataset/README.md for manual download instructions."
        ) from exc


def load_ai4i(path: str = config.AI4I_RAW) -> pd.DataFrame:
    """Load the raw AI4I 2020 Predictive Maintenance dataset."""
    path = ensure_ai4i(path)
    df = pd.read_csv(path)
    # Normalise headers (strip stray whitespace/BOM)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    logger.info("Loaded AI4I dataset: %s rows x %s cols", df.shape[0], df.shape[1])
    return df


def load_cmapss(path: str = config.CMAPSS_TRAIN) -> pd.DataFrame:
    """Load a NASA C-MAPSS FD001 file (train or test) into a tidy DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"C-MAPSS file not found at {path}. "
            "Download it from NASA's Prognostics Data Repository "
            "(https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) "
            "or see dataset/README.md - the time-series module will not run without it."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    logger.info("Loaded C-MAPSS file %s: %s rows, %s units",
                os.path.basename(path), df.shape[0], df["unit"].nunique())
    return df


def load_cmapss_rul(path: str = config.CMAPSS_RUL) -> pd.Series:
    """Load the RUL vector for the C-MAPSS test set (cycles remaining at last cycle)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"C-MAPSS RUL file not found at {path}.")
    return pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])["rul"]


def add_rul_column(train_df: pd.DataFrame) -> pd.DataFrame:
    """Append a Remaining-Useful-Life (RUL) column computed per engine unit."""
    df = train_df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def dataset_summary() -> dict:
    """Return a small metadata dict for reporting / the dashboard."""
    summary = {
        "ai4i": {
            "path": config.AI4I_RAW,
            "available": os.path.exists(config.AI4I_RAW),
            "source": "UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset",
            "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
        },
        "cmapss_fd001": {
            "train_path": config.CMAPSS_TRAIN,
            "test_path": config.CMAPSS_TEST,
            "rul_path": config.CMAPSS_RUL,
            "available": os.path.exists(config.CMAPSS_TRAIN),
            "source": "NASA Prognostics Center of Excellence - CMAPSS Turbofan Engine Degradation Simulation Data Set (FD001)",
            "url": "https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/",
        },
    }
    if summary["ai4i"]["available"]:
        df = load_ai4i()
        summary["ai4i"].update({
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "failure_rate": float(df[config.TARGET_COLUMN].mean()),
        })
    return summary
