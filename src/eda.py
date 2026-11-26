"""
Exploratory data analysis - generates every chart required by the spec and
saves them as PNG files under outputs/visualizations/.
"""
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend - safe on servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
PALETTE = {0: "#2e86ab", 1: "#d64541"}


def _save(fig, name: str) -> str:
    path = os.path.join(config.VIZ_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path)
    return path


def chart_dataset_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    axes[0].bar(["Normal (0)", "Failure (1)"], counts.values, color=[PALETTE[0], PALETTE[1]])
    axes[0].set_title("Target Class Counts")
    axes[0].set_ylabel("Rows")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}\n({100*v/len(df):.2f}%)", ha="center", va="bottom", fontsize=11)
    axes[0].set_ylim(0, counts.max() * 1.15)
    type_counts = df["Type"].value_counts() if "Type" in df.columns else pd.Series(dtype=int)
    if len(type_counts):
        axes[1].pie(type_counts.values, labels=[f"Type {k}" for k in type_counts.index],
                    autopct="%1.1f%%", colors=["#f6bd60", "#84a59d", "#f28482"], startangle=90)
        axes[1].set_title("Product Type Distribution")
    fig.suptitle("Dataset Overview Distribution", fontweight="bold")
    return _save(fig, "dataset_distribution.png")


def chart_failure_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x=config.TARGET_COLUMN, hue=config.TARGET_COLUMN, data=df, ax=axes[0],
                  palette=[PALETTE[0], PALETTE[1]], legend=False)
    axes[0].set_title("Machine Failure Distribution")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Normal", "Failure"])
    mode_cols = [c for c in config.FAILURE_MODE_COLUMNS if c in df.columns]
    mode_counts = df[mode_cols].sum().sort_values()
    axes[1].barh(mode_counts.index, mode_counts.values, color="#6d6875")
    axes[1].set_title("Failure Mode Counts (TWF/HDF/PWF/OSF/RNF)")
    axes[1].set_xlabel("Occurrences")
    fig.suptitle("Failure Analysis", fontweight="bold")
    return _save(fig, "failure_distribution.png")


def chart_temperature_analysis(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["Air temperature [K]"], kde=True, ax=axes[0], color=PALETTE[0])
    axes[0].set_title("Air Temperature")
    sns.histplot(df["Process temperature [K]"], kde=True, ax=axes[1], color="#e07a5f")
    axes[1].set_title("Process Temperature")
    sns.boxplot(x=config.TARGET_COLUMN, y=df["Process temperature [K]"] - df["Air temperature [K]"],
                hue=config.TARGET_COLUMN, data=df, ax=axes[2], palette=[PALETTE[0], PALETTE[1]], legend=False)
    axes[2].set_xlabel("Machine failure"); axes[2].set_ylabel("ΔT (K)")
    axes[2].set_title("Temperature Difference by Failure")
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["Normal", "Failure"])
    fig.suptitle("Temperature Analysis", fontweight="bold")
    return _save(fig, "temperature_analysis.png")


def chart_rotational_speed(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Rotational speed [rpm]"], kde=True, ax=axes[0], color="#3d5a80")
    axes[0].set_title("Rotational Speed Distribution")
    sns.boxplot(x=config.TARGET_COLUMN, y="Rotational speed [rpm]", hue=config.TARGET_COLUMN, data=df,
                ax=axes[1], palette=[PALETTE[0], PALETTE[1]], legend=False)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Normal", "Failure"])
    axes[1].set_title("Rotational Speed by Failure")
    fig.suptitle("Rotational Speed Analysis", fontweight="bold")
    return _save(fig, "rotational_speed_analysis.png")


def chart_torque_analysis(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Torque [Nm]"], kde=True, ax=axes[0], color="#ee6c4d")
    axes[0].set_title("Torque Distribution")
    sns.boxplot(x=config.TARGET_COLUMN, y="Torque [Nm]", hue=config.TARGET_COLUMN, data=df,
                ax=axes[1], palette=[PALETTE[0], PALETTE[1]], legend=False)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Normal", "Failure"])
    axes[1].set_title("Torque by Failure")
    fig.suptitle("Torque Analysis", fontweight="bold")
    return _save(fig, "torque_analysis.png")


def chart_tool_wear(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Tool wear [min]"], kde=True, ax=axes[0], color="#2a9d8f")
    axes[0].set_title("Tool Wear Distribution")
    sns.boxplot(x=config.TARGET_COLUMN, y="Tool wear [min]", hue=config.TARGET_COLUMN, data=df,
                ax=axes[1], palette=[PALETTE[0], PALETTE[1]], legend=False)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Normal", "Failure"])
    axes[1].set_title("Tool Wear by Failure")
    fig.suptitle("Tool Wear Analysis", fontweight="bold")
    return _save(fig, "tool_wear_analysis.png")


def chart_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Numeric Features)", fontweight="bold")
    return _save(fig, "correlation_heatmap.png")


def chart_sensor_relationships(df: pd.DataFrame) -> str:
    sample = df.sample(n=min(1500, len(df)), random_state=config.RANDOM_STATE)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    sns.scatterplot(x="Torque [Nm]", y="Rotational speed [rpm]", hue=config.TARGET_COLUMN,
                    data=sample, palette=PALETTE, alpha=0.6, ax=axes[0], s=25)
    axes[0].set_title("Torque vs Rotational Speed")
    sns.scatterplot(x="Air temperature [K]", y="Process temperature [K]", hue=config.TARGET_COLUMN,
                    data=sample, palette=PALETTE, alpha=0.6, ax=axes[1], s=25)
    axes[1].set_title("Air vs Process Temperature")
    sns.scatterplot(x="Tool wear [min]", y="Torque [Nm]", hue=config.TARGET_COLUMN,
                    data=sample, palette=PALETTE, alpha=0.6, ax=axes[2], s=25)
    axes[2].set_title("Tool Wear vs Torque")
    for ax in axes:
        ax.legend(title="Failure", labels=["Normal", "Failure"])
    fig.suptitle("Sensor Relationship Analysis", fontweight="bold")
    return _save(fig, "sensor_relationships.png")


def chart_failure_feature_comparisons(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    compare_cols = config.RAW_NUMERIC
    for ax, col in zip(axes.flat, compare_cols):
        sns.boxplot(x=config.TARGET_COLUMN, y=col, hue=config.TARGET_COLUMN, data=df,
                    ax=ax, palette=[PALETTE[0], PALETTE[1]], legend=False)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Failure"])
        ax.set_title(f"{col} by Failure")
    ax = axes.flat[len(compare_cols)]
    tmp = df.copy()
    tmp["temp_difference"] = tmp["Process temperature [K]"] - tmp["Air temperature [K]"]
    sns.violinplot(x=config.TARGET_COLUMN, y="temp_difference", hue=config.TARGET_COLUMN, data=tmp,
                   ax=ax, palette=[PALETTE[0], PALETTE[1]], legend=False)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Failure"])
    ax.set_title("Temperature Difference by Failure")
    fig.suptitle("Failure-related Feature Comparisons", fontweight="bold")
    return _save(fig, "feature_comparisons.png")


def run_eda(df: pd.DataFrame) -> list:
    charts = [
        chart_dataset_distribution(df), chart_failure_distribution(df),
        chart_temperature_analysis(df), chart_rotational_speed(df),
        chart_torque_analysis(df), chart_tool_wear(df),
        chart_correlation_heatmap(df), chart_sensor_relationships(df),
        chart_failure_feature_comparisons(df),
    ]
    logger.info("EDA complete - %d charts saved", len(charts))
    return charts
