"""
feature_engineering.py
═══════════════════════════════════════════════════════
Transform raw 18-column HVAC batch into engineered features
for anomaly detection. Produces a 15-feature vector for the
Isolation Forest model.

Feature design rationale:
  - Rolling stats detect drift (not just point anomalies)
  - Z-scores normalize across sensors with different units
  - Physics-derived cross-sensor features capture causality
  - Persistence flags directly feed Layer 1 rule engine
  - Time context filters normal startup/defrost artifacts

Context window (previous_tail) is critical:
  Without the last 30 rows of the previous batch,
  rolling windows at batch boundaries would be NaN
  or incorrectly calculated from only the current batch.
═══════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from datetime import datetime

from carrier_baseline import CARRIER_BASELINE, get_z_score

# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────

# Sensors to compute rolling statistics for (key diagnostic signals)
ROLLING_SENSORS = [
    "suction_pressure_bar",
    "coil_delta_T_C",
    "superheat_C",
    "chilled_water_flow_LPS",
    "discharge_pressure_bar",
    "blower_current_A",
    "compressor_current_A",
]

# Sensors to compute rate-of-change for (drift detection)
DELTA_SENSORS = [
    "suction_pressure_bar",
    "coil_delta_T_C",
    "discharge_pressure_bar",
    "chilled_water_flow_LPS",
]

# Rolling windows in minutes (= rows, since 1 row/min)
WINDOWS = [15, 30]

# Final 15-feature vector for Isolation Forest input
MODEL_FEATURES = [
    "z_suction_pressure_bar",
    "z_coil_delta_T_C",
    "z_superheat_C",
    "z_chilled_water_flow_LPS",
    "z_discharge_pressure_bar",
    "z_blower_current_A",
    "suction_pressure_bar_delta_15",
    "coil_delta_T_C_delta_15",
    "superheat_instability",          # = rolling_std_15 of superheat
    "compression_ratio",
    "coil_efficiency_index",
    "system_COP",
    "pressure_differential",          # discharge - suction
    "critical_persistence_suction_pressure_bar",   # count of |z|>3 in last 5 rows
    "is_business_hours",
]


# ─────────────────────────────────────────────────────
# MAIN ENGINEERING FUNCTION
# ─────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    previous_tail: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Transform raw sensor batch into full enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Current batch (60 rows, 18+ sensor columns)
    previous_tail : pd.DataFrame or None
        Last 30 rows of the previous batch.
        Required for accurate rolling features at batch boundaries.
        Pass None for the first batch (will use batch-internal context only).

    Returns
    -------
    pd.DataFrame
        Input df + all engineered features (columns appended).
        Row count equals input df row count.
    """
    # Combine previous tail + current batch for rolling calculation
    if previous_tail is not None and len(previous_tail) > 0:
        combined = pd.concat([previous_tail, df], ignore_index=True)
        tail_len = len(previous_tail)
    else:
        combined = df.copy()
        tail_len = 0

    # Ensure timestamp is parsed for time features
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    combined = _add_rolling_features(combined)
    combined = _add_delta_features(combined)
    combined = _add_z_scores(combined)
    combined = _add_physics_features(combined)
    combined = _add_persistence_flags(combined)
    combined = _add_time_context(combined)

    # Return only the current batch rows (strip the tail context)
    result = combined.iloc[tail_len:].copy()
    result = result.reset_index(drop=True)
    return result


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group A: Rolling statistics.
    Only rolling_std_15 and rolling_mean_30 go into the final feature
    vector to avoid dimensionality explosion (7 sensors × 2 windows ×
    4 stats = 56 columns otherwise).
    """
    for sensor in ROLLING_SENSORS:
        if sensor not in df.columns:
            continue
        s = df[sensor]
        for window in WINDOWS:
            roll = s.rolling(window=window, min_periods=1)
            df[f"{sensor}_rolling_mean_{window}"]  = roll.mean()
            df[f"{sensor}_rolling_std_{window}"]   = roll.std().fillna(0)
            df[f"{sensor}_rolling_min_{window}"]   = roll.min()
    return df


def _add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group B: Rate of change features.
    delta_5min and delta_15min detect drift direction and speed.
    suction_pressure rate_of_change (bar/min) is the primary leak signal.
    """
    for sensor in DELTA_SENSORS:
        if sensor not in df.columns:
            continue
        s = df[sensor]
        df[f"{sensor}_delta_5"]  = s.diff(5).fillna(0)
        df[f"{sensor}_delta_15"] = s.diff(15).fillna(0)

    # Explicit per-minute rate (for display in tickets)
    if "suction_pressure_bar" in df.columns:
        df["suction_pressure_rate_of_change"] = (
            df["suction_pressure_bar"].diff(1).fillna(0)
        )
    return df


def _add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group C: Z-scores vs Carrier 30XA baseline for all 13 raw sensors.
    Signed z (negative = below mean) — rule engine uses abs(z).
    """
    for sensor in CARRIER_BASELINE.keys():
        if sensor not in df.columns:
            continue
        b = CARRIER_BASELINE[sensor]
        df[f"z_{sensor}"] = (df[sensor] - b["mean"]) / b["std"]
    return df


def _add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group D: Cross-sensor physics features.
    These encode thermodynamic relationships that single-sensor
    z-scores miss (e.g., pressure differential diagnostic).
    """
    # Superheat instability: rolling std of superheat over 15 min
    # High std → EXV hunting; this is the primary EXV malfunction signal
    if "superheat_C_rolling_std_15" in df.columns:
        df["superheat_instability"] = df["superheat_C_rolling_std_15"]
    elif "superheat_C" in df.columns:
        df["superheat_instability"] = (
            df["superheat_C"].rolling(15, min_periods=1).std().fillna(0)
        )

    # Pressure differential: rises with condenser fouling, drops with low charge
    if "discharge_pressure_bar" in df.columns and "suction_pressure_bar" in df.columns:
        df["pressure_differential"] = (
            df["discharge_pressure_bar"] - df["suction_pressure_bar"]
        )

    # Thermal efficiency: how much coil delta_T per unit chilled water delta_T
    # Drops when coil is iced (reduced heat transfer surface)
    if "coil_delta_T_C" in df.columns and "chilled_water_delta_T_C" in df.columns:
        df["thermal_efficiency_ratio"] = (
            df["coil_delta_T_C"] / (df["chilled_water_delta_T_C"] + 1e-6)
        )

    return df


def _add_persistence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group E: Persistence flags for Layer 1 rule engine.

    Counts how many of the last 5 rows crossed advisory (|z|>2)
    or critical (|z|>3) thresholds for each key sensor.

    These directly implement the 3-of-5 persistence filter:
    Layer 1 reads advisory_persistence and critical_persistence
    columns to determine if the anomaly is sustained vs transient.
    """
    key_sensors = [
        "suction_pressure_bar",
        "coil_delta_T_C",
        "superheat_C",
        "discharge_pressure_bar",
        "chilled_water_flow_LPS",
    ]

    for sensor in key_sensors:
        z_col = f"z_{sensor}"
        if z_col not in df.columns:
            continue

        abs_z = df[z_col].abs()

        # Count of last 5 rows with |z| > 2.0 (advisory threshold)
        df[f"advisory_persistence_{sensor}"] = (
            (abs_z > 2.0)
            .rolling(window=5, min_periods=1)
            .sum()
            .astype(int)
        )

        # Count of last 5 rows with |z| > 3.0 (critical threshold)
        df[f"critical_persistence_{sensor}"] = (
            (abs_z > 3.0)
            .rolling(window=5, min_periods=1)
            .sum()
            .astype(int)
        )

    return df


def _add_time_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group F: Time context features.
    Helps model distinguish anomalies from normal startup artifacts.

    is_defrost_cycle: detected from coil_delta_T sudden rise pattern
    (defrost causes temporary coil delta_T spike — NOT a freeze signal).
    """
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour_of_day"]      = ts.dt.hour
        df["is_business_hours"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 18)).astype(int)
        # Startup window: 6-8 AM — sensor readings less stable during ramp-up
        df["startup_window"]   = ((ts.dt.hour >= 6) & (ts.dt.hour < 8)).astype(int)
    else:
        df["hour_of_day"]       = 12
        df["is_business_hours"] = 1
        df["startup_window"]    = 0

    # Defrost cycle detection: coil_delta_T rises suddenly (>2°C in 5 min)
    # During defrost, freeze-like signals are expected — mark to suppress false alarms
    if "coil_delta_T_C" in df.columns:
        coil_rise = df["coil_delta_T_C"].diff(5).fillna(0)
        df["is_defrost_cycle"] = (coil_rise > 2.0).astype(int)
    else:
        df["is_defrost_cycle"] = 0

    return df


# ─────────────────────────────────────────────────────
# MODEL FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────

def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract exactly the 15-column feature vector for Isolation Forest input.

    Missing columns are filled with 0 (safe default — represents normal).
    Column order is fixed to match the trained scaler's expected input.
    """
    feature_df = pd.DataFrame(index=df.index)
    for feat in MODEL_FEATURES:
        if feat in df.columns:
            feature_df[feat] = df[feat].fillna(0)
        else:
            feature_df[feat] = 0.0
            # Warn only in development; in production this should be logged
    return feature_df[MODEL_FEATURES]  # Enforce column order


# ─────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("FEATURE ENGINEERING — Self-test")
    print("=" * 60)

    # Generate synthetic test data
    from hvac_simulator import generate_dataset
    import glob

    generate_dataset(total_minutes=120, output_dir="data/raw",
                     modes=["low_refrigerant"], seed=7)

    files = sorted(glob.glob("data/raw/low_refrigerant/*.parquet"))
    if not files:
        print("No parquet files found — run hvac_simulator.py first")
        sys.exit(1)

    raw_df = pd.read_parquet(files[0])
    print(f"\nRaw batch: {len(raw_df)} rows × {len(raw_df.columns)} columns")

    enriched = engineer_features(raw_df)
    print(f"Enriched:  {len(enriched)} rows × {len(enriched.columns)} columns")

    model_feats = get_model_features(enriched)
    print(f"Model features: {len(model_feats.columns)} columns")
    print(f"  Columns: {list(model_feats.columns)}")
    print(f"\nSample (last 3 rows):")
    print(model_feats.tail(3).to_string(float_format="{:.3f}".format))

    # Test cross-batch continuity
    if len(files) > 1:
        raw2   = pd.read_parquet(files[1])
        tail   = enriched.tail(30)
        enr2        = engineer_features(raw2, previous_tail=tail)
        model_feats2 = get_model_features(enr2)
        print(f"\nCross-batch test: {len(enr2)} rows, "
              f"NaN count: {model_feats2.isna().sum().sum()}")

    print("\n✓ feature_engineering.py — OK")
