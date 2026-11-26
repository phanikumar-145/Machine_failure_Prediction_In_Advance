"""
Train & compare classification models for failure prediction.

Models: Logistic Regression (baseline, class-balanced, scaled),
        Random Forest (class-balanced, randomised hyperparameter search),
        XGBoost (scale_pos_weight, randomised hyperparameter search).

Class imbalance is handled via class weights / scale_pos_weight.
Outputs: trained Joblib models, model_metadata.json, feature_importance.csv,
         sample_failure_predictions.csv (real test-set outputs).
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import engineer_features, make_model_matrix
from src.preprocessing import clean_pipeline

logger = logging.getLogger(__name__)


def _build_models(y_train: pd.Series) -> dict:
    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    scale_pos = neg / pos
    logger.info("Class balance -> neg=%d pos=%d scale_pos_weight=%.2f", neg, pos, scale_pos)

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                   random_state=config.RANDOM_STATE)),
    ])
    rf = RandomForestClassifier(class_weight="balanced_subsample",
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(scale_pos_weight=scale_pos, random_state=config.RANDOM_STATE,
                        n_jobs=-1, eval_metric="logloss", tree_method="hist")
    return {"logistic_regression": logreg, "random_forest": rf, "xgboost": xgb}


def _tune(model_name, model, X_train, y_train):
    """Small randomised search - practical for a 10k-row dataset, keeps runtime sane."""
    if model_name == "random_forest":
        space = {"n_estimators": [200, 300, 400],
                 "max_depth": [6, 10, 14, None],
                 "min_samples_leaf": [1, 2, 4],
                 "max_features": ["sqrt", "log2"]}
    elif model_name == "xgboost":
        space = {"n_estimators": [200, 300, 400],
                 "max_depth": [3, 5, 7],
                 "learning_rate": [0.03, 0.08, 0.15],
                 "subsample": [0.7, 0.9, 1.0],
                 "colsample_bytree": [0.7, 0.9, 1.0]}
    elif model_name == "logistic_regression":
        space = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        return model
    search = RandomizedSearchCV(model, space, n_iter=min(8, _space_size(space)),
                                scoring="f1", cv=3, n_jobs=-1,
                                random_state=config.RANDOM_STATE, refit=True)
    search.fit(X_train, y_train)
    logger.info("[%s] best CV f1=%.4f params=%s", model_name, search.best_score_, search.best_params_)
    return search.best_estimator_


def _space_size(space: dict) -> int:
    size = 1
    for v in space.values():
        size *= len(v)
    return size


def train_all(df_engineered: pd.DataFrame | None = None) -> dict:
    if df_engineered is None:
        if os.path.exists(config.ENGINEERED_FEATURES):
            df_engineered = pd.read_csv(config.ENGINEERED_FEATURES)
        else:
            df_engineered = engineer_features(clean_pipeline())

    X = make_model_matrix(df_engineered)
    y = df_engineered[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    logger.info("Model matrix: %d features -> %s", X.shape[1], list(X.columns))

    fitted = {}
    for name, model in _build_models(y_train).items():
        logger.info("Training %s ...", name)
        fitted[name] = _tune(name, model, X_train, y_train)

    # Persist models with the exact filenames the ZIP/spec requires
    joblib.dump(fitted["random_forest"], os.path.join(config.MODELS_DIR, "random_forest.pkl"))
    joblib.dump(fitted["xgboost"], os.path.join(config.MODELS_DIR, "xgboost_model.pkl"))
    joblib.dump(fitted["logistic_regression"], os.path.join(config.MODELS_DIR, "logistic_regression.pkl"))

    # Feature importance (RF + XGB native; LR via |coefficients|)
    importance = pd.DataFrame({"feature": config.MODEL_FEATURES})
    importance["importance_random_forest"] = fitted["random_forest"].feature_importances_
    importance["importance_xgboost"] = fitted["xgboost"].feature_importances_
    importance["importance_logistic_regression"] = np.abs(
        fitted["logistic_regression"].named_steps["clf"].coef_[0])
    importance = importance.sort_values("importance_random_forest", ascending=False).reset_index(drop=True)
    importance.to_csv(os.path.join(config.REPORTS_DIR, "feature_importance.csv"), index=False)

    # Sample predictions on the held-out test set (real model outputs)
    proba = fitted["xgboost"].predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    sample = X_test.copy()
    sample["actual_failure"] = y_test.values
    sample["predicted_failure"] = preds
    sample["failure_probability"] = proba.round(6)
    sample["risk_level"] = pd.cut(proba, bins=[-0.01, config.RISK_LOW, config.RISK_MEDIUM,
                                               config.RISK_HIGH, 1.01],
                                  labels=["Low", "Medium", "High", "Critical"])
    sample.to_csv(os.path.join(config.PREDICTIONS_DIR, "sample_failure_predictions.csv"), index=False)

    metadata = {
        "project": "Predictive Maintenance Using Machine Learning and Time-Series Analysis",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "AI4I 2020 Predictive Maintenance Dataset (UCI)",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "features": list(X.columns),
        "raw_feature_names": config.MODEL_FEATURES,
        "feature_name_cleaning": "XGBoost-safe: brackets removed and spaces replaced with '_' (see config.clean_feature_name)",
        "target": config.TARGET_COLUMN,
        "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE,
        "models": {
            "logistic_regression": "logistic_regression.pkl",
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost_model.pkl",
        },
        "class_imbalance_handling": "class_weight='balanced' / 'balanced_subsample' (LR, RF); scale_pos_weight (XGB)",
        "metrics_file": "outputs/reports/evaluation_summary.json",
    }
    with open(config.MODEL_METADATA, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return {"fitted": fitted, "X_test": X_test, "y_test": y_test,
            "X_train": X_train, "y_train": y_train, "importance": importance}
