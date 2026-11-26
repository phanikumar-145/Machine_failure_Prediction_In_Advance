"""
Time-series analysis & forecasting on NASA C-MAPSS FD001 (real run-to-failure data).

Method
------
1. Load FD001 training trajectories (100 engines, one operating condition).
2. Per engine unit, order by cycle (chronological) and build lag features
   (lag 1..4, causal rolling mean/std) - strictly past information.
3. Chronological split per unit: first 80% of cycles = train, remaining 20% = test.
4. Train one GradientBoostingRegressor per selected sensor (top-4 by variance,
   near-constant sensors excluded).
5. Evaluate on the held-out tail with MAE / RMSE / MAPE (epsilon-guarded).
6. Produce a genuine multi-step recursive forecast for the NEXT 20 cycles for a
   reference engine, persisted to outputs/predictions/forecasting_results.csv.
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset.data_loader import load_cmapss, add_rul_column

logger = logging.getLogger(__name__)

N_LAGS = 4
HORIZON = 20          # cycles ahead for the recursive future forecast
MAX_UNITS_EVAL = 100  # use all units


def select_sensors(df: pd.DataFrame, n: int = 4) -> list:
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    variances = df[sensor_cols].var().sort_values(ascending=False)
    chosen = [s for s in variances.index if variances[s] > 1e-3][:n]
    logger.info("Selected sensors for forecasting: %s", chosen)
    return chosen


def build_supervised(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Per-unit chronological supervised table with past-only lag features."""
    frames = []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        s = g[sensor]
        out = pd.DataFrame({
            "unit": unit,
            "cycle": g["cycle"],
            f"{sensor}_lag1": s.shift(1),
            f"{sensor}_lag2": s.shift(2),
            f"{sensor}_lag3": s.shift(3),
            f"{sensor}_lag4": s.shift(4),
            f"{sensor}_roll_mean_4": s.shift(1).rolling(4, min_periods=1).mean(),
            f"{sensor}_roll_std_4": s.shift(1).rolling(4, min_periods=1).std().fillna(0.0),
            "op_setting_1": g["op_setting_1"],
            "op_setting_2": g["op_setting_2"],
            "op_setting_3": g["op_setting_3"],
            "target": s,
        })
        frames.append(out)
    sup = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    return sup


def feature_columns(sensor: str) -> list:
    return ([f"{sensor}_lag{i}" for i in range(1, N_LAGS + 1)]
            + [f"{sensor}_roll_mean_4", f"{sensor}_roll_std_4",
               "op_setting_1", "op_setting_2", "op_setting_3", "cycle"])


def chronological_split(sup: pd.DataFrame, frac: float = 0.8):
    """Split per unit by cycle order - chronological, no shuffling, no leakage."""
    train_idx, test_idx = [], []
    for unit, idx in sup.groupby("unit").groups.items():
        idx = list(idx)
        cut = int(len(idx) * frac)
        train_idx.extend(idx[:cut]); test_idx.extend(idx[cut:])
    return sup.loc[train_idx], sup.loc[test_idx]


def _mape(y_true, y_pred) -> float:
    denom = np.maximum(np.abs(y_true), 1e-3)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def recursive_forecast(model, history: list, op_row: pd.Series, start_cycle: int,
                       sensor: str, horizon: int = HORIZON) -> list:
    """Multi-step ahead forecast feeding predictions back as lags."""
    hist = list(history[-N_LAGS:])
    preds = []
    roll_hist = list(history)
    for step in range(1, horizon + 1):
        lags = hist[::-1]  # lag1 = most recent
        feat = {f"{sensor}_lag{i+1}": lags[i] for i in range(N_LAGS)}
        feat[f"{sensor}_roll_mean_4"] = float(np.mean(roll_hist[-4:]))
        feat[f"{sensor}_roll_std_4"] = float(np.std(roll_hist[-4:]))
        feat["op_setting_1"] = float(op_row.get("op_setting_1", 0.0))
        feat["op_setting_2"] = float(op_row.get("op_setting_2", 0.0))
        feat["op_setting_3"] = float(op_row.get("op_setting_3", 0.0))
        feat["cycle"] = start_cycle + step
        X = pd.DataFrame([feat])[feature_columns(sensor)]
        yhat = float(model.predict(X)[0])
        preds.append(yhat)
        hist = (hist + [yhat])[-N_LAGS:]
        roll_hist.append(yhat)
    return preds


def run_forecasting() -> dict:
    df = add_rul_column(load_cmapss(config.CMAPSS_TRAIN))
    sensors = select_sensors(df, n=4)

    metrics, models, forecast_rows, plot_data = {}, {}, [], {}
    for sensor in sensors:
        sup = build_supervised(df, sensor)
        train_df, test_df = chronological_split(sup, frac=0.8)
        feats = feature_columns(sensor)
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                          max_depth=3, random_state=config.RANDOM_STATE)
        model.fit(train_df[feats], train_df["target"])
        y_pred = model.predict(test_df[feats])
        y_true = test_df["target"].values
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = _mape(y_true, y_pred)
        metrics[sensor] = {"mae": mae, "rmse": rmse, "mape_pct": mape,
                           "n_train": int(len(train_df)), "n_test": int(len(test_df))}
        logger.info("[%s] MAE=%.4f RMSE=%.4f MAPE=%.2f%% (train=%d test=%d)",
                    sensor, mae, rmse, mape, len(train_df), len(test_df))
        models[sensor] = model
        for (u, c, a, p) in zip(test_df["unit"], test_df["cycle"], y_true, y_pred):
            forecast_rows.append({"unit": int(u), "cycle": int(c), "sensor": sensor,
                                  "actual": round(float(a), 5), "predicted": round(float(p), 5),
                                  "row_type": "test_evaluation"})
        plot_data[sensor] = (test_df, y_pred)

    # ---- genuine multi-step recursive forecast for a reference engine ----
    ref_unit = int(df.groupby("unit")["cycle"].max().idxmin())  # shortest trajectory - cleanest demo
    ref = df[df["unit"] == ref_unit].sort_values("cycle")
    future_forecasts = {}
    for sensor in sensors:
        history = ref[sensor].tolist()
        preds = recursive_forecast(models[sensor], history,
                                   ref.iloc[-1], int(ref["cycle"].max()), sensor, HORIZON)
        future_forecasts[sensor] = preds
        for i, p in enumerate(preds, start=1):
            forecast_rows.append({"unit": ref_unit, "cycle": int(ref["cycle"].max()) + i,
                                  "sensor": sensor, "actual": np.nan,
                                  "predicted": round(float(p), 5),
                                  "row_type": "future_forecast"})

    results_df = pd.DataFrame(forecast_rows)
    results_df.to_csv(os.path.join(config.PREDICTIONS_DIR, "forecasting_results.csv"), index=False)

    # Persist forecasting bundle
    joblib.dump({"models": models, "sensors": sensors, "n_lags": N_LAGS,
                 "feature_columns": {s: feature_columns(s) for s in sensors}},
                os.path.join(config.MODELS_DIR, "forecasting_model.pkl"))

    overall = {
        "dataset": "NASA C-MAPSS FD001 (train_FD001.txt, run-to-failure)",
        "method": "GradientBoostingRegressor on cycle + op settings + past-only lag(1-4) & causal rolling stats",
        "split": "chronological 80/20 per engine unit (no shuffle, no leakage)",
        "horizon_future_forecast_cycles": HORIZON,
        "reference_engine_unit": ref_unit,
        "sensors": metrics,
        "overall": {
            "mae": float(np.mean([m["mae"] for m in metrics.values()])),
            "rmse": float(np.mean([m["rmse"] for m in metrics.values()])),
            "mape_pct": float(np.mean([m["mape_pct"] for m in metrics.values()])),
        },
    }
    with open(os.path.join(config.REPORTS_DIR, "forecasting_metrics.json"), "w") as fh:
        json.dump(overall, fh, indent=2)

    _plot_forecast(plot_data, ref, future_forecasts, sensors, ref_unit)
    _plot_trends(df, sensors)
    logger.info("Forecasting complete. Overall MAE=%.4f RMSE=%.4f MAPE=%.2f%%",
                overall["overall"]["mae"], overall["overall"]["rmse"], overall["overall"]["mape_pct"])
    return overall


def _plot_forecast(plot_data, ref, future_forecasts, sensors, ref_unit):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, sensor in zip(axes.flat, sensors):
        test_df, y_pred = plot_data[sensor]
        for unit in test_df["unit"].unique()[:40]:
            u = test_df[test_df["unit"] == unit].sort_values("cycle")
            ax.plot(u["cycle"], u["target"], color="#adb5bd", lw=0.6, alpha=0.5)
        ref_test = test_df[test_df["unit"] == ref_unit].sort_values("cycle")
        if len(ref_test):
            pred_map = dict(zip(test_df["cycle"], y_pred))
            ax.plot(ref_test["cycle"], ref_test["target"], "o-", ms=4, color="#d64541",
                    label=f"Actual (unit {ref_unit})")
            ax.plot(ref_test["cycle"], [pred_map[c] for c in ref_test["cycle"]], "s--", ms=4,
                    color="#2e86ab", label="Predicted")
        ax.set_title(sensor); ax.set_xlabel("Cycle"); ax.set_ylabel("Sensor value")
        ax.legend(fontsize=9)
    fig.suptitle("C-MAPSS FD001 - Actual vs Predicted Sensor Values (held-out final 20% of cycles)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.VIZ_DIR, "time_series_forecast.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_trends(df: pd.DataFrame, sensors: list):
    units = df["unit"].unique()[:8]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, sensor in zip(axes.flat, sensors):
        for u in units:
            g = df[df["unit"] == u].sort_values("cycle")
            ax.plot(g["cycle"], g[sensor], alpha=0.75, lw=1.2, label=f"unit {u}")
        ax.set_title(f"{sensor} degradation trend (sample units)")
        ax.set_xlabel("Cycle"); ax.set_ylabel(sensor)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("C-MAPSS FD001 - Sensor Degradation Trends", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.VIZ_DIR, "cmapss_sensor_trends.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
