"""
Model evaluation: classification metrics, comparison, plots, JSON/CSV reports.

Best model is chosen by F1-score (tie-break: failure-class recall) - the two
metrics that matter most for a safety/maintenance use case with rare positives.
"""
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0,
                                                       output_dict=True),
    }
    return metrics


def evaluate_models(fitted: dict, X_test, y_test) -> dict:
    results = {}
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = classification_metrics(y_test, y_pred, y_proba)
        logger.info("[%s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f", name,
                    results[name]["accuracy"], results[name]["precision"],
                    results[name]["recall"], results[name]["f1_score"], results[name]["roc_auc"])

    comparison = pd.DataFrame([
        {"model": name,
         "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
         "f1_score": m["f1_score"], "roc_auc": m["roc_auc"]}
        for name, m in results.items()
    ]).sort_values(["f1_score", "recall"], ascending=False).reset_index(drop=True)
    comparison.to_csv(os.path.join(config.REPORTS_DIR, "model_comparison.csv"), index=False)

    best_name = comparison.iloc[0]["model"]
    summary = {
        "best_model": best_name,
        "selection_criteria": "highest F1-score (tie-break: failure-class recall)",
        "metrics": results,
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(os.path.join(config.REPORTS_DIR, "classification_report.json"), "w") as fh:
        json.dump(summary["metrics"][best_name], fh, indent=2)
    with open(os.path.join(config.REPORTS_DIR, "evaluation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    plot_confusion_matrices(fitted, X_test, y_test)
    plot_roc_curves(fitted, X_test, y_test)
    plot_model_comparison(comparison)

    # Record best model in metadata
    meta_path = config.MODEL_METADATA
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["best_model"] = best_name
        meta["best_model_metrics"] = {k: v for k, v in results[best_name].items()
                                      if k not in ("classification_report", "confusion_matrix")}
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
    return summary


def plot_confusion_matrices(fitted: dict, X_test, y_test):
    fig, axes = plt.subplots(1, len(fitted), figsize=(6 * len(fitted), 5))
    if len(fitted) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Failure"]); ax.set_yticklabels(["Normal", "Failure"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontweight="bold")
    fig.suptitle("Confusion Matrices (Test Set)", fontweight="bold")
    fig.savefig(os.path.join(config.VIZ_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(fitted: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, model in fitted.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name.replace('_', ' ').title()} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Failure Class", fontweight="bold")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(config.VIZ_DIR, "roc_curve.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison))
    width = 0.15
    colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, comparison[metric], width, label=metric.replace("_", " ").title(), color=color)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in comparison["model"]])
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Test Set)", fontweight="bold")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.savefig(os.path.join(config.VIZ_DIR, "model_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame):
    top = importance.sort_values("importance_random_forest", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_random_forest"], color="#2a9d8f", label="Random Forest")
    ax.barh(top["feature"], top["importance_xgboost"], color="#e76f51", alpha=0.55, label="XGBoost")
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance (RF & XGB)", fontweight="bold")
    ax.legend()
    fig.savefig(os.path.join(config.VIZ_DIR, "feature_importance.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
