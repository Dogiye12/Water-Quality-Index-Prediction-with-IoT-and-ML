#!/usr/bin/env python3
"""
Water-Quality-Index-Prediction-with-IoT-and-ML (Synthetic Data Demo)
--------------------------------------------------------------------
This script simulates IoT sensor data for surface water monitoring and
trains ML models to predict a Water Quality Index (WQI). It creates
> 1,000 synthetic records (configurable), trains/evaluates models, and
exports artifacts (CSV, model, and plots).

Dependencies:
    pip install numpy pandas scikit-learn matplotlib joblib

Run:
    python water_quality_iot_ml.py --n 1500 --seed 42

Outputs (in ./outputs/):
    - water_quality_iot.csv                (synthetic dataset)
    - wqi_model_random_forest.joblib       (trained model)
    - metrics.json                         (evaluation metrics)
    - feature_importance.png               (model feature importance)
    - residuals_plot.png                   (predicted vs observed residuals)

Author: ChatGPT
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------- Synthetic Generator ----------------------------

@dataclass
class Ranges:
    pH: Tuple[float, float] = (6.0, 9.0)
    temp_c: Tuple[float, float] = (18.0, 34.0)
    tds_mgL: Tuple[float, float] = (50.0, 1200.0)
    turbidity_ntu: Tuple[float, float] = (0.1, 150.0)
    cond_uScm: Tuple[float, float] = (50.0, 2000.0)
    do_mgL: Tuple[float, float] = (1.0, 12.0)
    bod_mgL: Tuple[float, float] = (0.5, 30.0)
    cod_mgL: Tuple[float, float] = (5.0, 200.0)
    nitrate_mgL: Tuple[float, float] = (0.1, 50.0)
    ammonia_mgL: Tuple[float, float] = (0.01, 10.0)
    orp_mV: Tuple[float, float] = (-150.0, 450.0)


def _rand_uniform(n, low, high, rng):
    return rng.uniform(low, high, size=n)


def generate_synthetic_iot(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rg = Ranges()

    clusters = rng.choice(["upstream", "midstream", "downstream"], size=n, p=[0.35, 0.45, 0.20])

    pH = _rand_uniform(n, *rg.pH, rng)
    temp_c = _rand_uniform(n, *rg.temp_c, rng)
    tds = _rand_uniform(n, *rg.tds_mgL, rng)
    turb = _rand_uniform(n, *rg.turbidity_ntu, rng)
    cond = _rand_uniform(n, *rg.cond_uScm, rng)
    do = _rand_uniform(n, *rg.do_mgL, rng)
    bod = _rand_uniform(n, *rg.bod_mgL, rng)
    cod = _rand_uniform(n, *rg.cod_mgL, rng)
    no3 = _rand_uniform(n, *rg.nitrate_mgL, rng)
    nh3 = _rand_uniform(n, *rg.ammonia_mgL, rng)
    orp = _rand_uniform(n, *rg.orp_mV, rng)

    cond = 0.7 * (tds / 1.5) + 0.3 * cond
    cod = 0.6 * (bod * 4.0 + 10) + 0.4 * cod
    turb = 0.5 * turb + 0.5 * (no3 * 2.0) + rng.normal(0, 2.0, n)

    for i, c in enumerate(clusters):
        if c == "downstream":
            tds[i] *= rng.uniform(1.1, 1.3)
            turb[i] *= rng.uniform(1.2, 1.4)
            bod[i] *= rng.uniform(1.2, 1.5)
            do[i] *= rng.uniform(0.7, 0.9)
            no3[i] *= rng.uniform(1.2, 1.5)
            nh3[i] *= rng.uniform(1.2, 1.6)
            orp[i] -= rng.uniform(20, 60)
        elif c == "upstream":
            tds[i] *= rng.uniform(0.8, 0.95)
            turb[i] *= rng.uniform(0.8, 0.95)
            bod[i] *= rng.uniform(0.7, 0.9)
            do[i] *= rng.uniform(1.05, 1.2)
            orp[i] += rng.uniform(10, 40)

    df = pd.DataFrame({
        "site_id": [f"S{i:04d}" for i in range(n)],
        "cluster": clusters,
        "pH": np.clip(pH, 5.5, 9.5),
        "temperature_C": np.clip(temp_c, 10, 40),
        "TDS_mgL": np.clip(tds, 10, 5000),
        "turbidity_NTU": np.clip(turb, 0.05, 500),
        "conductivity_uScm": np.clip(cond, 5, 8000),
        "DO_mgL": np.clip(do, 0.1, 14),
        "BOD_mgL": np.clip(bod, 0.1, 60),
        "COD_mgL": np.clip(cod, 1, 600),
        "nitrate_mgL": np.clip(no3, 0.01, 100),
        "ammonia_mgL": np.clip(nh3, 0.001, 40),
        "ORP_mV": np.clip(orp, -300, 600),
    })

    def si_ph(x): return np.clip(100 - np.maximum(0, np.abs(x - 7.4) - 1.1) * 12, 0, 100)
    def si_do(x): return np.clip((x - 3) / (12 - 3) * 100, 0, 100)
    def si_bod(x): return np.clip(100 - (x - 1.5) * 5.0, 0, 100)
    def si_turb(x): return np.clip(100 - (x / 2.0), 0, 100)
    def si_tds(x): return np.clip(100 - (x / 25.0), 0, 100)
    def si_no3(x): return np.clip(100 - (x * 3.0), 0, 100)
    def si_nh3(x): return np.clip(100 - (x * 8.0), 0, 100)
    def si_orp(x): return np.clip((x + 150) / (600 + 150) * 100, 0, 100)

    weights = {"pH": 0.12, "DO": 0.18, "BOD": 0.16, "Turb": 0.12,
               "TDS": 0.10, "NO3": 0.10, "NH3": 0.12, "ORP": 0.10}

    wqi = (
        weights["pH"] * si_ph(df["pH"].values) +
        weights["DO"] * si_do(df["DO_mgL"].values) +
        weights["BOD"] * si_bod(df["BOD_mgL"].values) +
        weights["Turb"] * si_turb(df["turbidity_NTU"].values) +
        weights["TDS"] * si_tds(df["TDS_mgL"].values) +
        weights["NO3"] * si_no3(df["nitrate_mgL"].values) +
        weights["NH3"] * si_nh3(df["ammonia_mgL"].values) +
        weights["ORP"] * si_orp(df["ORP_mV"].values)
    )
    wqi = wqi - (df["temperature_C"].values - 20) * 0.6
    wqi = np.clip(wqi + np.random.normal(0, 3.5, size=n), 0, 100)

    df["WQI"] = wqi.round(2)
    df["WQI_class"] = pd.cut(df["WQI"], bins=[-0.1, 25, 50, 70, 90, 100.1],
                             labels=["Very Poor", "Poor", "Moderate", "Good", "Excellent"])
    return df


def train_and_evaluate(df: pd.DataFrame, seed: int = 42) -> Dict[str, float]:
    features = [
        "pH", "temperature_C", "TDS_mgL", "turbidity_NTU", "conductivity_uScm",
        "DO_mgL", "BOD_mgL", "COD_mgL", "nitrate_mgL", "ammonia_mgL", "ORP_mV"
    ]
    X = df[features].values
    y = df["WQI"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(n_estimators=400, random_state=seed, max_depth=None, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    dump(model, os.path.join(OUTPUT_DIR, "wqi_model_random_forest.joblib"))

    fig, ax = plt.subplots(figsize=(8, 5))
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names = np.array(features)[order]
    vals = importances[order]
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(0, linestyle="--")
    ax2.set_xlabel("Predicted WQI")
    ax2.set_ylabel("Residual (Observed - Predicted)")
    ax2.set_title("Residuals vs Predicted")
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "residuals_plot.png"), dpi=150)
    plt.close(fig2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Water Quality Index Prediction with IoT and ML (Synthetic Demo)")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n <= 100:
        raise SystemExit("Please provide --n > 100")

    df = generate_synthetic_iot(args.n, seed=args.seed)
    csv_path = os.path.join(OUTPUT_DIR, "water_quality_iot.csv")
    df.to_csv(csv_path, index=False)

    metrics = train_and_evaluate(df, seed=args.seed)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
