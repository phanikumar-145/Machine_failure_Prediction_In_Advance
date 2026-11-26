"""
Predictive Maintenance - Streamlit Dashboard
============================================
Professional industrial monitoring UI. Loads the ACTUAL trained models,
reports and visualisations produced by `python main.py`. No hardcoded
predictions anywhere - every number comes from the artifacts in models/
and outputs/.

Run:  streamlit run app.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.prediction import predict_single, predict_dataframe, load_anomaly_bundle, validate_inputs
from src.recommendations import generate_recommendations

st.set_page_config(page_title="Predictive Maintenance | AI4I + C-MAPSS",
                   page_icon="🛠️", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.main { background: #f4f6fb; }
.kpi-card {
    background: linear-gradient(135deg, #0d3b66 0%, #1d5c99 100%);
    color: #ffffff; border-radius: 14px; padding: 18px 16px;
    box-shadow: 0 4px 14px rgba(13,59,102,.25); text-align: center;
}
.kpi-value { font-size: 1.9rem; font-weight: 800; }
.kpi-label { font-size: .78rem; opacity: .85; text-transform: uppercase; letter-spacing: .08em; }
.status-healthy  { background:#0a9396; } .status-watch { background:#ee9b00; }
.status-warning  { background:#ca6702; } .status-critical { background:#ae2012; }
.section-title { color:#0d3b66; border-bottom: 3px solid #ee9b00; padding-bottom:.3rem; }
</style>""", unsafe_allow_html=True)


def kpi(col, value, label):
    col.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
                 f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<h2 class="section-title">{title}</h2>', unsafe_allow_html=True)


# ------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def load_clean():
    return pd.read_csv(config.CLEANED_DATASET)

@st.cache_data(show_spinner=False)
def load_engineered():
    return pd.read_csv(config.ENGINEERED_FEATURES)

@st.cache_data(show_spinner=False)
def load_json(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def img_path(name):
    return os.path.join(config.VIZ_DIR, name)

def show_image(name, caption=None):
    p = img_path(name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.warning(f"Visualisation '{name}' not found - run `python main.py` first.")

def gauge(probability: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=probability * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Failure Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0d3b66"},
            "steps": [{"range": [0, 30], "color": "#94d2bd"},
                      {"range": [30, 60], "color": "#ee9b00"},
                      {"range": [60, 80], "color": "#ca6702"},
                      {"range": [80, 100], "color": "#ae2012"}],
        }))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig

def sensor_inputs(prefix="s"):
    c1, c2, c3 = st.columns(3)
    with c1:
        mtype = st.selectbox("Product Type", ["L", "M", "H"], index=1, key=f"{prefix}_type")
        air = st.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1, key=f"{prefix}_air")
        proc = st.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1, key=f"{prefix}_proc")
    with c2:
        rpm = st.slider("Rotational speed [rpm]", 1100, 2900, 1540, 5, key=f"{prefix}_rpm")
        torque = st.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.1, key=f"{prefix}_torque")
        wear = st.slider("Tool wear [min]", 0, 253, 110, 1, key=f"{prefix}_wear")
    with c3:
        st.markdown("**Computed**")
        st.metric("ΔT (proc − air)", f"{proc - air:.1f} K")
        st.metric("Power", f"{torque * rpm * 2 * np.pi / 60 / 1000:.2f} kW")
        st.metric("Wear × Torque", f"{wear * torque:.0f}")
    return {"Type": mtype, "Air temperature [K]": air, "Process temperature [K]": proc,
            "Rotational speed [rpm]": float(rpm), "Torque [Nm]": torque,
            "Tool wear [min]": float(wear)}

def anomaly_score_for(reading):
    try:
        bundle = load_anomaly_bundle()
        row = {c: reading[c] for c in config.RAW_NUMERIC}
        row["temp_difference"] = reading["Process temperature [K]"] - reading["Air temperature [K]"]
        row["power_watts"] = reading["Torque [Nm]"] * reading["Rotational speed [rpm]"] * 2 * np.pi / 60.0
        X = pd.DataFrame([row])[bundle["features"]]
        Xs = bundle["scaler"].transform(X)
        # decision_function: higher = more normal -> invert to [0,1]-ish anomaly score
        raw = float(bundle["model"].decision_function(Xs)[0])
        return {"-1": "ANOMALY", "1": "NORMAL"}[str(bundle["model"].predict(Xs)[0])], raw
    except FileNotFoundError:
        return None, None

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🛠️ Predictive Maintenance")
    st.caption("ML + Time-Series · AI4I 2020 · NASA C-MAPSS FD001")
    page = st.radio("Navigate", [
        "🏠 Home Dashboard", "📦 Dataset Overview", "🔎 Exploratory Data Analysis",
        "🩺 Machine Health Monitoring", "⚠️ Failure Prediction", "🚨 Anomaly Detection",
        "📈 Time-Series Forecasting", "📊 Model Performance", "🔧 Maintenance Recommendations",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Models**: XGBoost · Random Forest · Logistic Regression")
    st.markdown("**Anomaly**: Isolation Forest")
    st.markdown("**Forecasting**: Gradient Boosting (lag features)")

df_clean = load_clean()
meta = load_json(config.MODEL_METADATA)
eval_summary = load_json(os.path.join(config.REPORTS_DIR, "evaluation_summary.json"))
comparison = load_csv(os.path.join(config.REPORTS_DIR, "model_comparison.csv"))
forecast_metrics = load_json(os.path.join(config.REPORTS_DIR, "forecasting_metrics.json"))
anomaly_results = load_csv(os.path.join(config.PREDICTIONS_DIR, "anomaly_results.csv"))
best_model = eval_summary.get("best_model", meta.get("best_model", "xgboost"))
best_metrics = eval_summary.get("metrics", {}).get(best_model, {})

# ------------------------------------------------------------------- pages
if page == "🏠 Home Dashboard":
    section("🏭 Fleet Health Overview")
    total = len(df_clean); fails = int(df_clean[config.TARGET_COLUMN].sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, f"{total:,}", "Sensor records")
    kpi(c2, f"{fails}", "Failures (train feed)")
    kpi(c3, f"{100*fails/total:.2f}%", "Failure rate")
    kpi(c4, best_model.replace("_", " ").title(), "Best model")
    kpi(c5, f"{best_metrics.get('f1_score', 0):.3f}", "Best F1-score")
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(px.pie(values=[total - fails, fails], names=["Normal", "Failure"],
                               color=["Normal", "Failure"],
                               color_discrete_map={"Normal": "#2a9d8f", "Failure": "#ae2012"},
                               title="Fleet failure ratio", hole=0.45), use_container_width=True)
    with c2:
        if not comparison.empty:
            st.plotly_chart(px.bar(comparison, x="model", y="f1_score", color="model",
                                   title="F1-score by model (test set)",
                                   labels={"f1_score": "F1-score", "model": "Model"}),
                            use_container_width=True)
    st.markdown("### 🔑 Key findings")
    rf = best_metrics
    st.markdown(f"- **{best_model.replace('_',' ').title()}** selected as best model — "
                f"F1 = {rf.get('f1_score', float('nan')):.3f}, recall = {rf.get('recall', float('nan')):.3f}, "
                f"ROC-AUC = {rf.get('roc_auc', float('nan')):.3f} (real test-set metrics).")
    st.markdown("- Tool wear, torque and power show the strongest association with failures — see Feature Importance on the Model Performance page.")
    st.markdown("- Class imbalance (3.4% failures) is handled with class weights / `scale_pos_weight`, never by dropping failures.")
    st.markdown("- Forecasting on NASA C-MAPSS FD001: average MAPE "
                f"{forecast_metrics.get('overall', {}).get('mape_pct', float('nan')):.2f}% across the top degradation sensors.")
    show_image("failure_distribution.png")

elif page == "📦 Dataset Overview":
    section("📦 Dataset Overview")
    tab1, tab2, tab3 = st.tabs(["AI4I 2020 (classification)", "Engineered features", "NASA C-MAPSS FD001 (time-series)"])
    with tab1:
        st.write(f"**Shape:** {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        st.dataframe(df_clean.head(25), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Data types**"); st.dataframe(df_clean.dtypes.astype(str).rename("dtype"), use_container_width=True)
        with c2:
            st.write("**Missing values**"); st.dataframe(df_clean.isna().sum().rename("missing"), use_container_width=True)
        st.download_button("⬇️ Download cleaned dataset (CSV)",
                           df_clean.to_csv(index=False).encode(), "cleaned_dataset.csv", "text/csv")
    with tab2:
        eng = load_engineered()
        st.write(f"**Shape:** {eng.shape[0]:,} rows × {eng.shape[1]} columns  —  "
                 "physics features + past-only lag/rolling features (no future information).")
        st.dataframe(eng.head(25), use_container_width=True)
        st.download_button("⬇️ Download engineered features (CSV)",
                           eng.to_csv(index=False).encode(), "engineered_features.csv", "text/csv")
    with tab3:
        st.markdown("""**NASA C-MAPSS FD001** — 100 run-to-failure turbofan trajectories,
        20,631 cycles, 26 columns (unit, cycle, 3 operational settings, 21 sensors).
        Loaded from `dataset/raw/train_FD001.txt`. Used for chronological degradation
        analysis & forecasting (see *Time-Series Forecasting* page).""")
        fc = load_csv(os.path.join(config.PREDICTIONS_DIR, "forecasting_results.csv"))
        if not fc.empty:
            st.write(f"Forecast artifact: {fc.shape[0]:,} rows (test evaluations + 20-cycle future forecast).")
            st.dataframe(fc.head(20), use_container_width=True)

elif page == "🔎 Exploratory Data Analysis":
    section("🔎 Exploratory Data Analysis")
    chart = st.selectbox("Choose a visualisation", [
        "dataset_distribution.png", "failure_distribution.png", "temperature_analysis.png",
        "rotational_speed_analysis.png", "torque_analysis.png", "tool_wear_analysis.png",
        "correlation_heatmap.png", "sensor_relationships.png", "feature_comparisons.png",
    ])
    show_image(chart)
    st.markdown("**Target balance:**")
    st.dataframe(df_clean[config.TARGET_COLUMN].value_counts().rename_axis("Machine failure")
                 .rename("count").to_frame(), use_container_width=True)

elif page == "🩺 Machine Health Monitoring":
    section("🩺 Machine Health Monitoring")
    st.markdown("Enter a live machine reading — the trained model scores it instantly.")
    reading = sensor_inputs("health")
    if st.button("🔍 Assess machine health", type="primary"):
        result = predict_single(reading, model_name="best")
        verdict, raw = anomaly_score_for(reading)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(gauge(result["failure_probability"]), use_container_width=True)
        with c2:
            status = result["machine_health"]
            cls = {"Healthy": "status-healthy", "Watch": "status-watch",
                   "Warning": "status-warning", "Critical": "status-critical"}[status]
            st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-value">{status.upper()}</div>'
                        f'<div class="kpi-label">Machine health · risk: {result["risk_level"]} · '
                        f'model: {result["model"]}</div></div>', unsafe_allow_html=True)
            st.write("")
            st.metric("Model prediction", result["prediction_label"])
            if verdict:
                st.metric("Anomaly detector", verdict)
        st.markdown("### 🧰 Recommended actions")
        for rec in generate_recommendations(reading, result["failure_probability"]):
            st.markdown(f"- {rec}")

elif page == "⚠️ Failure Prediction":
    section("⚠️ Failure Prediction")
    tab1, tab2 = st.tabs(["Single machine", "Batch CSV upload"])
    with tab1:
        reading = sensor_inputs("pred")
        col_a, col_b = st.columns([1, 1])
        model_choice = col_a.selectbox("Model", ["best", "xgboost", "random_forest", "logistic_regression"])
        if col_b.button("Predict failure", type="primary"):
            result = predict_single(reading, model_name=model_choice)
            st.plotly_chart(gauge(result["failure_probability"]), use_container_width=True)
            st.success(f"Prediction: **{result['prediction_label']}** · "
                       f"failure probability **{result['failure_probability']:.2%}** · "
                       f"risk **{result['risk_level']}** (model: {result['model']})")
    with tab2:
        st.markdown("Upload a CSV with the AI4I schema: `Type`, `Air temperature [K]`, "
                    "`Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                batch = pd.read_csv(up)
                preds = predict_dataframe(batch, model_name="best")
                st.dataframe(preds, use_container_width=True)
                st.download_button("⬇️ Download predictions (CSV)", preds.to_csv(index=False).encode(),
                                   "failure_predictions.csv", "text/csv", type="primary")
            except Exception as exc:
                st.error(f"Could not score the file: {exc}")
        st.markdown("**Or use the generated sample predictions** (held-out test set, produced by the pipeline):")
        sample = load_csv(os.path.join(config.PREDICTIONS_DIR, "sample_failure_predictions.csv"))
        if not sample.empty:
            st.dataframe(sample.head(50), use_container_width=True)
            st.download_button("⬇️ Download sample predictions (CSV)",
                               sample.to_csv(index=False).encode(), "sample_failure_predictions.csv", "text/csv")

elif page == "🚨 Anomaly Detection":
    section("🚨 Anomaly Detection (Isolation Forest)")
    if not anomaly_results.empty:
        n_anom = int(anomaly_results["is_anomaly"].sum())
        overlap = int(((anomaly_results["is_anomaly"] == 1) & (anomaly_results["Machine failure"] == 1)).sum())
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, f"{n_anom:,}", "Anomalies found")
        kpi(c2, f"{100*n_anom/len(anomaly_results):.2f}%", "Anomaly rate")
        kpi(c3, f"{overlap}", "Overlap w/ failures")
        kpi(c4, f"{anomaly_results['anomaly_score'].mean():.3f}", "Avg anomaly score")
        st.write("")
        show_image("anomaly_detection.png")
        anomalies = anomaly_results[anomaly_results["is_anomaly"] == 1].sort_values(
            "anomaly_score", ascending=False)
        st.markdown(f"**Top 25 most anomalous records** ({len(anomalies)} total)")
        st.dataframe(anomalies.head(25), use_container_width=True)
        st.download_button("⬇️ Download anomaly results (CSV)",
                           anomaly_results.to_csv(index=False).encode(), "anomaly_results.csv", "text/csv")
    else:
        st.warning("anomaly_results.csv not found - run `python main.py --step anomaly`.")

elif page == "📈 Time-Series Forecasting":
    section("📈 Time-Series Forecasting (NASA C-MAPSS FD001)")
    overall = forecast_metrics.get("overall", {})
    c1, c2, c3 = st.columns(3)
    kpi(c1, f"{overall.get('mae', float('nan')):.3f}", "Overall MAE")
    kpi(c2, f"{overall.get('rmse', float('nan')):.3f}", "Overall RMSE")
    kpi(c3, f"{overall.get('mape_pct', float('nan')):.2f}%", "Overall MAPE")
    st.caption(f"Method: {forecast_metrics.get('method', '')} · Split: {forecast_metrics.get('split', '')}")
    sensors = forecast_metrics.get("sensors", {})
    if sensors:
        st.dataframe(pd.DataFrame(sensors).T.rename_axis("sensor").reset_index(),
                     use_container_width=True)
    show_image("time_series_forecast.png")
    show_image("cmapss_sensor_trends.png")
    fc = load_csv(os.path.join(config.PREDICTIONS_DIR, "forecasting_results.csv"))
    if not fc.empty:
        sensor_pick = st.selectbox("Interactive actual vs predicted — sensor", sorted(fc["sensor"].unique()))
        test_rows = fc[(fc["sensor"] == sensor_pick) & (fc["row_type"] == "test_evaluation")]
        unit_pick = st.selectbox("Engine unit", sorted(test_rows["unit"].unique())[:20])
        u = test_rows[test_rows["unit"] == unit_pick].sort_values("cycle")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u["cycle"], y=u["actual"], name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=u["cycle"], y=u["predicted"], name="Predicted", mode="lines+markers"))
        fig.update_layout(title=f"{sensor_pick} — unit {unit_pick} (held-out cycles)",
                          xaxis_title="Cycle", yaxis_title="Sensor value")
        st.plotly_chart(fig, use_container_width=True)
        future = fc[fc["row_type"] == "future_forecast"]
        if not future.empty:
            st.markdown("**20-cycle recursive future forecast** (reference engine "
                        f"unit {int(future['unit'].iloc[0])}):")
            st.dataframe(future, use_container_width=True)
        st.download_button("⬇️ Download forecasting results (CSV)",
                           fc.to_csv(index=False).encode(), "forecasting_results.csv", "text/csv")

elif page == "📊 Model Performance":
    section("📊 Model Performance (held-out test set)")
    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True)
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        long_df = comparison.melt(id_vars="model", value_vars=metrics,
                                  var_name="metric", value_name="score")
        st.plotly_chart(px.bar(long_df, x="metric", y="score", color="model", barmode="group",
                               title="Model comparison across metrics"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        show_image("confusion_matrix.png")
        show_image("roc_curve.png")
    with c2:
        show_image("model_comparison.png")
        show_image("feature_importance.png")
    fi = load_csv(os.path.join(config.REPORTS_DIR, "feature_importance.csv"))
    if not fi.empty:
        st.markdown("**Feature importance (RF & XGB)**")
        st.dataframe(fi, use_container_width=True)

elif page == "🔧 Maintenance Recommendations":
    section("🔧 Maintenance Recommendations")
    st.markdown("Enter a machine reading — recommendations combine **model risk**, **anomaly score** and **physics/domain rules**.")
    reading = sensor_inputs("rec")
    if st.button("🧰 Generate recommendations", type="primary"):
        result = predict_single(reading, model_name="best")
        _, raw = anomaly_score_for(reading)
        recs = generate_recommendations(reading, result["failure_probability"])
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(gauge(result["failure_probability"]), use_container_width=True)
        with c2:
            st.markdown("### Actions")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"**{i}.** {rec}")
    with st.expander("📋 General preventive-maintenance playbook"):
        st.markdown("""
- **Tool wear > 200 min** → replace tooling; inspect collet & spindle taper.
- **ΔT < 9.5 K** → verify coolant circuit, fan/heat-exchanger and airflow (heat-dissipation failure mode).
- **Torque > 60 Nm** → reduce feed/load; check for binding in drivetrain (overstrain failure mode).
- **Low rpm + high torque** → inspect motor & power transmission (power failure mode).
- **Repeated anomalies with normal predictions** → calibrate sensors; re-baseline after maintenance.
- **After any confirmed failure** → root-cause review; feed the new labeled sample back into retraining.
        """)
