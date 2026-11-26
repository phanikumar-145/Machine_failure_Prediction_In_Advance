import re
"""
Central configuration for the Predictive Maintenance ML project.

All paths, constants and feature definitions live here so every module
(training, evaluation, prediction, dashboard) stays consistent.
"""
import os

# ---------------------------------------------------------------- paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
VIZ_DIR = os.path.join(OUTPUTS_DIR, "visualizations")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
PROCESSED_DIR = os.path.join(OUTPUTS_DIR, "processed_data")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for _d in [RAW_DIR, VIZ_DIR, REPORTS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------- datasets ----
AI4I_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
AI4I_RAW = os.path.join(RAW_DIR, "ai4i2020.csv")
CMAPSS_DIR = RAW_DIR
CMAPSS_TRAIN = os.path.join(RAW_DIR, "train_FD001.txt")
CMAPSS_TEST = os.path.join(RAW_DIR, "test_FD001.txt")
CMAPSS_RUL = os.path.join(RAW_DIR, "RUL_FD001.txt")

# ------------------------------------------------------------ constants ----
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Machine failure"

# Raw (instantaneous) numeric sensor columns of the AI4I dataset
RAW_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_COLUMN = "Type"          # categorical: L / M / H
ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # never used as features (leakage)

# Physics-inspired engineered features (instantaneous -> safe for single-row prediction)
ENGINEERED_NUMERIC = [
    "temp_difference",
    "power_watts",
    "tool_wear_torque",
    "torque_per_rpm",
]

# Features the classification models are trained on (order matters).
MODEL_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC + ["Type_H", "Type_L", "Type_M"]

SENSOR_COLUMNS = RAW_NUMERIC  # used for rolling / lag style engineering on ordered data

# Failure-probability thresholds used for risk stratification
RISK_LOW = 0.30
RISK_MEDIUM = 0.60
RISK_HIGH = 0.80

# ----------------------------------------------------------- outputs ------
CLEANED_DATASET = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")
ENGINEERED_FEATURES = os.path.join(PROCESSED_DIR, "engineered_features.csv")
TRAINING_LOG = os.path.join(LOGS_DIR, "training.log")
MODEL_METADATA = os.path.join(MODELS_DIR, "model_metadata.json")


# XGBoost forbids '[', ']' and '<' in feature names. The training/inference
# code sanitises model-matrix column names with this single shared function
# so both sides always agree.
def clean_feature_name(name: str) -> str:
    return re.sub(r"[\[\]<>]", "", name).replace(" ", "_").replace("/", "_")
