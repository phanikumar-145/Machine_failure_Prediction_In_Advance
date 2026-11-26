# Predictive Maintenance Using Machine Learning and Time-Series Analysis

End-to-end Machine Learning project that analyses industrial equipment sensor
data, identifies abnormal operating patterns, engineers time-based features and
trains models that predict potential machine failures **before they happen** —
complete with an interactive Streamlit monitoring dashboard.

**Datasets used (real, publicly available):**

| Dataset | Role | Source |
|---|---|---|
| AI4I 2020 Predictive Maintenance Dataset (10,000 rows, 14 cols) | Failure classification | [UCI ML Repository #601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| NASA C-MAPSS FD001 Turbofan Engine Degradation (100 engines, 20,631 cycles) | Chronological degradation analysis & sensor forecasting | [NASA PCoE Prognostic Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |

Both raw files are **bundled** in this archive (`dataset/raw/`) — no download is
required to run the project. The AI4I file was downloaded from the UCI static
endpoint; the FD001 files are byte-identical copies of the official MATLAB
text exports obtained from a public mirror of the NASA repository.

---

## 1. Problem statement

Unplanned equipment breakdowns are among the costliest events in discrete
manufacturing and fleet operations. Traditional calendar/usage-based
preventive maintenance either over-services healthy machines or under-serves
degrading ones. This project builds a data-driven alternative:

> Given live sensor readings (temperatures, rotational speed, torque, tool
> wear), estimate the probability of an imminent failure, flag anomalous
> operating regimes, forecast sensor degradation trends, and translate those
> signals into concrete maintenance actions.

## 2. Objectives

1. Clean, validate and profile real industrial sensor data.
2. Engineer physically meaningful and time-aware features (past-only, no leakage).
3. Train and fairly compare Logistic Regression, Random Forest and XGBoost.
4. Detect anomalous operating points with Isolation Forest.
5. Forecast chronological sensor degradation on NASA C-MAPSS FD001.
6. Evaluate everything on a held-out test set — no invented numbers.
7. Ship a professional Streamlit dashboard bound to the real trained models.
8. Reduce unexpected breakdowns, maintenance cost and downtime.

## 3. Real-world use case

A machine shop monitors CNC spindles / milling machines. Each cycle the PLC
emits air/process temperature, spindle speed, torque and tool-wear counters.
This system scores every reading in near real time; machines crossing risk
thresholds are queued for inspection *before* the tool snaps or the bearing
seizes, converting emergency stops into planned 30-minute interventions.

## 4. Technologies

Python 3.10+ · NumPy · Pandas · Scikit-learn · XGBoost · Matplotlib · Seaborn
· Plotly · Streamlit · Joblib · (stats: SciPy) — see `requirements.txt`

## 5. Architecture

```
 raw CSVs ─► data_loader ─► preprocessing ─► feature_engineering ─┬─► train_model ─► evaluation
                                                                  ├─► anomaly_detection
                                                                  └─► forecasting (C-MAPSS)
                                   │
                     outputs/ (visualizations · reports · predictions · logs)
                                   │
                              app.py (Streamlit dashboard)
```

## 6. Folder structure

```
Predictive_Maintenance_ML/
├── app.py                    # Streamlit dashboard (9 pages)
├── main.py                   # Pipeline orchestrator (python main.py)
├── config.py                 # Paths, constants, feature definitions
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   ├── README.md             # Provenance & download instructions
│   ├── data_loader.py        # Loading + integrity helpers
│   └── raw/                  # ai4i2020.csv · train/test/RUL FD001
├── src/
│   ├── preprocessing.py      # Cleaning, validation, outlier analysis, split
│   ├── eda.py                # 9 EDA charts -> outputs/visualizations/
│   ├── feature_engineering.py# Physics + past-only lag/rolling features
│   ├── train_model.py        # LR/RF/XGB + tuning + persistence
│   ├── evaluation.py         # Metrics, comparison, ROC/CM/comparison plots
│   ├── anomaly_detection.py  # Isolation Forest + scoring + plots
│   ├── forecasting.py        # C-MAPSS chronological lag forecasting
│   ├── prediction.py         # Inference API (used by dashboard & tests)
│   └── recommendations.py    # Rule-based maintenance recommendations
├── models/                   # *.pkl (trained) + model_metadata.json
├── outputs/
│   ├── visualizations/       # 16 PNG charts (see §10)
│   ├── reports/              # classification_report.json, model_comparison.csv,
│   │                         # evaluation_summary.json, forecasting_metrics.json,
│   │                         # feature_importance.csv, outlier_analysis.csv, ...
│   ├── predictions/          # sample_failure_predictions.csv, anomaly_results.csv,
│   │                         # forecasting_results.csv
│   ├── processed_data/       # cleaned_dataset.csv, engineered_features.csv
│   └── logs/training.log
├── notebooks/exploratory_analysis.ipynb
└── tests/                    # pytest: preprocessing · features · prediction
```

## 7. Data preprocessing (`src/preprocessing.py`)

- **Missing values** → median (numeric) / mode (categorical). The raw AI4I file
  has none, but the path is implemented and unit-tested.
- **Duplicates** → dropped with count logged (0 found).
- **Outlier analysis** → IQR bounds computed per numeric feature and written to
  `outputs/reports/outlier_analysis.csv`. Rows are **not** deleted — extreme
  industrial readings carry failure signal.
- **Data-type validation** → numeric coercion, non-negative clips, categorical
  checks (`Type ∈ {L, M, H}`), integer target.
- **Feature selection** → failure-mode flags (TWF/HDF/PWF/OSF/RNF) and IDs are
  excluded from model features (they leak the target).
- **Split** → stratified 80/20 train/test (`random_state=42`), preserving the
  3.39% failure rate.

## 8. Feature engineering (`src/feature_engineering.py`)

**Instantaneous (physics-inspired, inference-safe):**
`temp_difference` (process − air), `power_watts` (τ·ω), `tool_wear_torque`
(wear × torque, overstrain proxy), `torque_per_rpm`.

**Chronological (past-only, zero look-ahead):** `*_lag1`, `*_diff`,
`*_roll_mean_5`, `*_roll_std_5` computed over UDI order for the sensor columns.
First-row lag falls back to the current reading (documented). These sequence
features are used for analysis, **not** for the classification model — training
the model on them would make single-row dashboard inference inconsistent.

**Wear progression:** `tool_wear_pct`, `tool_wear_stage` (0–3 bucket).

Full matrix: `outputs/processed_data/engineered_features.csv` (10,000 × 42).

## 9. Machine learning models (`src/train_model.py`)

| Model | Imbalance handling | Tuning |
|---|---|---|
| Logistic Regression (baseline, standardised pipeline) | `class_weight='balanced'` | RandomizedSearchCV on `C` (CV = 3, scoring = F1) |
| Random Forest | `class_weight='balanced_subsample'` | RandomizedSearchCV (n_estimators, max_depth, max_features, min_samples_leaf) |
| XGBoost | `scale_pos_weight = 7729/271 ≈ 28.5` | RandomizedSearchCV (depth, lr, subsample, colsample) |

**Actual held-out test results (2,000 rows, stratified):**

| Model | Accuracy | Precision | Recall (failure) | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.860 | 0.181 | 0.882 | 0.300 | 0.936 |
| Random Forest | 0.991 | 0.946 | 0.779 | 0.855 | 0.976 |
| **XGBoost (best)** | **0.9915** | **0.918** | **0.824** | **0.868** | **0.987** |

**Best model = XGBoost**, selected by highest F1 (tie-break recall); selection
rule is stored in `outputs/reports/evaluation_summary.json`. Logistic
Regression is kept as the recall-heavy safety net. All numbers are real
outputs of `python main.py`, reproducible with `random_state=42`.

## 10. Outputs (generated by executing the project)

Visualisations (`outputs/visualizations/`): `failure_distribution.png`,
`correlation_heatmap.png`, `temperature_analysis.png`, `torque_analysis.png`,
`rotational_speed_analysis.png`, `tool_wear_analysis.png`, `dataset_distribution.png`,
`sensor_relationships.png`, `feature_comparisons.png`, `feature_importance.png`,
`confusion_matrix.png`, `roc_curve.png`, `model_comparison.png`,
`anomaly_detection.png`, `time_series_forecast.png`, `cmapss_sensor_trends.png`.

Reports (`outputs/reports/`): `classification_report.json`,
`model_comparison.csv`, `evaluation_summary.json`, `forecasting_metrics.json`,
`feature_importance.csv`, `preprocessing_report.json`, `outlier_analysis.csv`.

Predictions (`outputs/predictions/`): `sample_failure_predictions.csv` (2,000
scored test rows), `anomaly_results.csv` (10,000 rows scored),
`forecasting_results.csv` (test-cycle predictions + 20-cycle future forecast).

Processed data: `cleaned_dataset.csv`, `engineered_features.csv`.
Logs: `outputs/logs/training.log`.

## 11. Time-series methodology (`src/forecasting.py`)

C-MAPSS FD001 has **no** timestamps either — the chronological axis is the
**cycle** (each unit is a run-to-failure trajectory). The pipeline:

1. Adds per-unit `RUL = max(cycle) − cycle`.
2. Selects the 4 highest-variance sensors (excludes near-constant ones).
3. Builds past-only features: cycle, 3 op settings, lag 1–4, causal rolling
   mean/std — **no shuffling, split is 80/20 per unit by cycle order**.
4. Trains one GradientBoostingRegressor per sensor.
5. Evaluates on the held-out final 20% of cycles (MAE / RMSE / MAPE).
6. Produces a genuine 20-cycle recursive multi-step forecast for a reference
   engine (feeds predictions back as lags).

**Actual results (FD001, test tail):**

| Sensor | MAE | RMSE | MAPE |
|---|---|---|---|
| sensor_9 | 8.93 | 17.38 | 0.10% |
| sensor_14 | 6.46 | 13.08 | 0.08% |
| sensor_4 | 5.74 | 7.11 | 0.40% |
| sensor_3 | 4.87 | 6.05 | 0.30% |
| **Overall** | **6.50** | **10.90** | **0.22%** |

Note: sensor values drift slowly relative to their magnitudes, so MAPE is low;
this is expected physics, not leakage — features are strictly past-only and the
split is chronological.

## 12. Anomaly detection (`src/anomaly_detection.py`)

IsolationForest (300 trees, `contamination=0.034` ≈ observed failure rate) over
sensor + engineered features, standardized. Per-row anomaly scores are
normalised to [0, 1] (1 = most anomalous). Actual run: **340 / 10,000 anomalies
(3.40%), 94 of which coincide with real failure events** — i.e. failures are
strongly over-represented among the anomalies. Outputs: `anomaly_model.pkl`,
`anomaly_detection.png` (RPM–torque space, PCA space, score histogram),
`outputs/predictions/anomaly_results.csv`.

## 13. Installation & execution

### Windows PowerShell

```powershell
# 1) Clone / unzip the project, then from the project root:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 2) (Optional) re-run the full pipeline — models/ and outputs/ are already generated
python main.py                 # full pipeline (preprocess→eda→train→anomaly→forecast)
python main.py --step train    # or a single stage

# 3) Launch the dashboard
streamlit run app.py           # opens http://localhost:8501

# 4) Run the unit tests
pytest tests/ -q
```

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py && streamlit run app.py && pytest tests/ -q
```

### Manual dataset download (only if `dataset/raw/` is wiped)

```bash
# AI4I 2020 (UCI)
curl -L -o dataset/raw/ai4i.zip "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
python -c "import zipfile; zipfile.ZipFile('dataset/raw/ai4i.zip').extractall('dataset/raw')"
# C-MAPSS FD001: fetch Damage Propagation Modeling.zip from the NASA PCoE
# Prognostic Data Repository (see dataset/README.md) and extract train_FD001.txt,
# test_FD001.txt, RUL_FD001.txt into dataset/raw/.
```

## 14. Dashboard

`streamlit run app.py` — 9 pages with sidebar navigation: Home Dashboard
(KPIs + fleet ratio), Dataset Overview (AI4I / engineered / C-MAPSS tabs,
CSV downloads), Exploratory Data Analysis, Machine Health Monitoring (live
input → probability gauge + health status + actions), Failure Prediction
(single machine or **batch CSV upload** with download), Anomaly Detection
(profile + top anomalies), Time-Series Forecasting (metrics, interactive
actual-vs-predicted per unit/sensor, 20-cycle recursive future forecast),
Model Performance (comparison, confusion matrices, ROC, feature importance),
Maintenance Recommendations (risk + anomaly + physics rules, plus a playbook).

## 15. Sample predictions

`outputs/predictions/sample_failure_predictions.csv` — first rows of the
held-out test set scored by XGBoost, e.g.

```
Type_M, failure_probability, predicted_failure, risk_level
   1          0.0004                0             Low
   1          0.7316                1             High
```

(exact columns: full feature vector + `actual_failure`, `predicted_failure`,
`failure_probability`, `risk_level`).

## 16. Limitations

- AI4I 2020 is a **snapshot** dataset (no timestamps); sequence features are
  approximated over file order and never fed to the classifier.
- Failure class is rare (3.39%); absolute precision is limited by design — F1 /
  recall are the honest selection criteria.
- C-MAPSS FD001 covers a single operating condition; FD002–FD004 (multiple
  conditions) would need condition-aware modeling.
- AI4I is lab-scale synthetic-but-physics-based data; on a real plant,
  retraining on site data with drift monitoring is mandatory.
- MAPE on slow-drifting sensors is low but can mislead for sensors near zero.

## 17. Future improvements

- Condition-aware FD002–FD004 models + RUL regression head (piecewise-linear RUL).
- Survival analysis (Cox / DeepSurv) and remaining-useful-life prediction API.
- SHAP-based per-prediction explanations inside the dashboard.
- Online learning / drift detection (Evidently) with automated retraining.
- MLflow model registry + CI (GitHub Actions) running the test suite.
- Edge deployment (ONNX) for on-machine inference at the PLC gateway.

## 18. Reproducibility

`random_state=42` everywhere; stratified split; deterministic pipelines.
Model files and metadata (`models/model_metadata.json`) record feature order,
training time, split sizes and the best-model selection rationale.

## 19. Dataset licenses & credits

AI4I 2020 © UCI ML Repository (CC BY 4.0). NASA PCoE CMAPSS — public NASA data;
credit: Saxena & Goebel, NASA Ames ( Damage Propagation Modeling, 2008).
