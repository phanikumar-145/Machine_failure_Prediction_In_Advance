"""
Data collection & preprocessing for the AI4I 2020 predictive-maintenance data.

Steps performed (all implemented, no placeholders):
  * loading (via dataset.data_loader)
  * missing-value handling (median for numeric, mode for categorical)
  * duplicate removal
  * outlier analysis (IQR rule; reported, not silently deleted - industrial
    extreme readings are exactly what we want the model to see)
  * data-type validation & coercion
  * target identification & sanity checks
  * stratified train/test separation
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset.data_loader import load_ai4i

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    df = df.copy()
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = df[col].median()
        else:
            fill = df[col].mode(dropna=True)
            fill = fill.iloc[0] if len(fill) else "Unknown"
        df[col] = df[col].fillna(fill)
        logger.info("Imputed %d missing value(s) in '%s' with %r", n_missing, col, fill)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate row(s)", before - len(df))
    return df


def validate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce expected dtypes; coerce numerics and fail loudly if impossible."""
    df = df.copy()
    for col in config.RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Tool wear [min]"] = df["Tool wear [min]"].clip(lower=0)
    df["Rotational speed [rpm]"] = df["Rotational speed [rpm]"].clip(lower=0)
    df["Torque [Nm]"] = df["Torque [Nm]"].clip(lower=0)
    for col in [config.TARGET_COLUMN] + config.FAILURE_MODE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df[config.TYPE_COLUMN] = df[config.TYPE_COLUMN].astype(str).str.strip().str.upper()
    bad = set(df[config.TYPE_COLUMN].unique()) - {"L", "M", "H"}
    if bad:
        raise ValueError(f"Unexpected Type value(s): {bad} (expected subset of L/M/H)")
    return df


def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """IQR-based outlier report for numeric columns. Does NOT drop rows."""
    rows = []
    for col in config.RAW_NUMERIC:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({
            "column": col, "q1": round(float(q1), 4), "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4), "lower_bound": round(float(low), 4),
            "upper_bound": round(float(high), 4),
            "n_outliers": n_out, "outlier_pct": round(100 * n_out / len(df), 3),
        })
    return pd.DataFrame(rows)


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the L/M/H product type (keeps LR well-behaved)."""
    dummies = pd.get_dummies(df[config.TYPE_COLUMN], prefix="Type", dtype=int)
    for col in ["Type_H", "Type_L", "Type_M"]:
        if col not in dummies.columns:
            dummies[col] = 0
    return pd.concat([df.drop(columns=[config.TYPE_COLUMN]), dummies[["Type_H", "Type_L", "Type_M"]]], axis=1)


def clean_pipeline(save_path: str = config.CLEANED_DATASET) -> pd.DataFrame:
    """Run the full cleaning stage and persist the cleaned dataset + report."""
    raw = load_ai4i()
    report = {"raw_rows": int(raw.shape[0]), "raw_columns": int(raw.shape[1])}

    df = handle_missing_values(raw)
    report["missing_values_total"] = int(raw.isna().sum().sum())
    df = remove_duplicates(df)
    report["duplicates_removed"] = int(report["raw_rows"] - len(df))
    df = validate_dtypes(df)

    outliers = analyze_outliers(df)
    outliers.to_csv(os.path.join(config.REPORTS_DIR, "outlier_analysis.csv"), index=False)
    report["outliers"] = outliers.to_dict(orient="records")

    report["clean_rows"] = int(df.shape[0])
    report["failure_rate"] = float(df[config.TARGET_COLUMN].mean())
    report["target_distribution"] = df[config.TARGET_COLUMN].value_counts().sort_index().to_dict()

    df.to_csv(save_path, index=False)
    with open(os.path.join(config.REPORTS_DIR, "preprocessing_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Cleaned dataset saved to %s (%d rows)", save_path, len(df))
    return df


def split_train_test(df: pd.DataFrame, target: str = config.TARGET_COLUMN):
    """Stratified train/test split. Returns (X_train_idx-based dfs, y...) as DataFrames."""
    feature_cols = [c for c in df.columns if c not in
                    [target, "UDI", "Product ID"] + config.FAILURE_MODE_COLUMNS]
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    logger.info("Train/test split: %d / %d rows (failure rate %.4f / %.4f)",
                len(X_train), len(X_test), y_train.mean(), y_test.mean())
    return X_train, X_test, y_train, y_test
