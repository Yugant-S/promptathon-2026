"""
anomaly_detector.py

Two-layer anomaly detection for HVAC freeze risk.
Now upgraded with Defrost Cycle Suppression.

Business Logic:
  - Defrost cycles cause transient sensor spikes (e.g. coil temp rise).
  - These are NOT faults but normal maintenance behavior.
  - The detector now filters out defrost/recovery rows before analysis.

FIX (defrost_recovery KeyError):
  Previously, check_batch accessed df["defrost_recovery"] directly
  without the same .get() guard used for "is_defrost_cycle".
  Any batch that arrived without this column (e.g. the first batch
  from a freshly started simulator, or data from an external source
  that doesn't include defrost metadata) raised a KeyError and halted
  the entire detection run.

  Both columns are now accessed via a consistent pattern:
    col in df.columns → .sum()   else → 0
  The analysis_df filter also guards both columns with the same
  pattern so a missing column never blocks analysis.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from carrier_baseline import (
    CARRIER_BASELINE, get_z_score, get_severity_from_z,
    PERSISTENCE_RULE, SEVERITY_ORDER, compare_severity,
)
from feature_engineering import engineer_features, get_model_features

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────

def _col_sum(df: pd.DataFrame, col: str) -> int:
    """Return sum of a boolean column, or 0 if the column does not exist."""
    return int(df[col].sum()) if col in df.columns else 0


def _filter_active_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows that are neither in a defrost cycle nor in the
    post-defrost recovery window.

    Handles the case where either column (or both) is absent from the
    DataFrame — in that case all rows are considered active.
    """
    mask = pd.Series(True, index=df.index)
    if "is_defrost_cycle" in df.columns:
        mask &= df["is_defrost_cycle"] == 0
    if "defrost_recovery" in df.columns:
        mask &= df["defrost_recovery"] == 0
    return df[mask]


# ─────────────────────────────────────────────────────
# LAYER 1 — Statistical Rule Engine
# ─────────────────────────────────────────────────────

class RuleEngine:
    """Statistical Rule Engine with Defrost Suppression."""

    def __init__(self):
        self.baseline = CARRIER_BASELINE
        self.sensors  = list(CARRIER_BASELINE.keys())

    def check_batch(self, df: pd.DataFrame) -> dict:
        """
        Analyse a batch, suppressing defrost / recovery rows.

        Defrost metadata columns (is_defrost_cycle, defrost_recovery)
        are optional.  Missing columns are treated as all-zero (no
        suppression), so batches from external data sources without
        defrost tracking still work correctly.
        """
        # ── Defrost suppression ───────────────────────────────────────────────
        defrost_rows  = _col_sum(df, "is_defrost_cycle")
        recovery_rows = _col_sum(df, "defrost_recovery")
        suppressed_rows = defrost_rows + recovery_rows
        active_rows     = len(df) - suppressed_rows

        if suppressed_rows > 0:
            print(
                f"  [DEFROST] {int(suppressed_rows)} rows suppressed "
                f"(defrost/recovery window)"
            )

        if active_rows == 0:
            return {
                "suppressed":           True,
                "suppressed_rows":      int(suppressed_rows),
                "reason":               "ENTIRE_BATCH_IN_DEFROST",
                "result":               "NORMAL",
                "worst_severity":       "NORMAL",
                "persistence_confirmed": False,
                "flagged_sensors":      [],
                "max_z_score":          0.0,
                "details":              {},
            }

        # ── Use only non-defrost rows for analysis ────────────────────────────
        analysis_df = _filter_active_rows(df)

        # ── Per-sensor z-score and persistence checks ─────────────────────────
        flagged_sensors  = []
        max_z            = 0.0
        worst_severity   = "NORMAL"
        details          = {}
        global_persist   = False

        for sensor in self.sensors:
            if sensor not in analysis_df.columns:
                continue

            z_series    = analysis_df[sensor].apply(lambda v: get_z_score(sensor, v))
            abs_z       = z_series.abs()
            max_row_z   = abs_z.max()

            # Persistence: 3-of-5 consecutive rows above advisory threshold
            n_rows = len(analysis_df)
            window = min(5, n_rows)
            persist = False
            for i in range(n_rows - window + 1):
                if (abs_z.iloc[i: i + window] > 2.0).sum() >= PERSISTENCE_RULE:
                    persist = True
                    break

            sensor_sev = get_severity_from_z(max_row_z)
            details[sensor] = {
                "max_z_score":  round(float(max_row_z), 4),
                "severity":     sensor_sev,
                "persist_3of5": persist,
            }

            if max_row_z > 2.0:
                flagged_sensors.append(sensor)
                if persist:
                    global_persist = True

            if max_row_z > max_z:
                max_z = float(max_row_z)

            worst_severity = compare_severity(worst_severity, sensor_sev)

        return {
            "flagged_sensors":       flagged_sensors,
            "max_z_score":           round(max_z, 4),
            "worst_severity":        worst_severity,
            "persistence_confirmed": global_persist,
            "details":               details,
            "suppressed":            False,
            "suppressed_rows":       int(suppressed_rows),
        }


# ─────────────────────────────────────────────────────
# LAYER 2 — Isolation Forest
# ─────────────────────────────────────────────────────

class IsolationForestDetector:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model     = None
        self.scaler    = None
        self.threshold = None

    def train(self, normal_df: pd.DataFrame) -> dict:
        enriched   = engineer_features(normal_df)
        feature_df = get_model_features(enriched).dropna()

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(feature_df.values)

        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        train_scores   = self.model.decision_function(X_scaled)
        # 5th-percentile threshold: only flags points more anomalous
        # than 95% of what was seen during normal operation.
        self.threshold = float(np.percentile(train_scores, 5))

        joblib.dump(self.model,  self.model_dir / "isolation_forest.pkl")
        joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
        with open(self.model_dir / "threshold.json", "w") as f:
            json.dump({"threshold": self.threshold}, f)

        return {"threshold": self.threshold}

    def load_model(self):
        self.model  = joblib.load(self.model_dir / "isolation_forest.pkl")
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        with open(self.model_dir / "threshold.json") as f:
            self.threshold = json.load(f)["threshold"]

    def predict(self, batch_df: pd.DataFrame, previous_tail=None) -> dict:
        if self.model is None:
            self.load_model()

        enriched   = engineer_features(batch_df, previous_tail=previous_tail)
        feature_df = get_model_features(enriched).fillna(0)
        X_scaled   = self.scaler.transform(feature_df.values)

        scores      = self.model.decision_function(X_scaled)
        is_anomaly  = scores < self.threshold
        anomaly_pct = float(is_anomaly.sum() / len(scores) * 100)

        # Require ≥ 15% of rows flagged before confirming.
        # Prevents a single outlier row from triggering IMMINENT_FAILURE
        # on an otherwise normal batch (was the original FIX-2).
        CONFIRM_THRESHOLD_PCT = 15.0

        return {
            "anomaly_scores": scores,
            "is_anomaly":     is_anomaly,
            "anomaly_pct":    round(anomaly_pct, 2),
            "worst_score":    round(float(scores.min()), 4),
            "mean_score":     round(float(scores.mean()), 4),
            "confirmed":      anomaly_pct >= CONFIRM_THRESHOLD_PCT,
        }


# ─────────────────────────────────────────────────────
# TOP-LEVEL DETECTION ENTRY POINT
# ─────────────────────────────────────────────────────

def detect_anomaly(
    batch_df,
    previous_tail=None,
    model_dir: str = "models",
) -> dict:
    """
    Run the two-layer detection pipeline on a single batch DataFrame.

    Decision logic:
      Layer 1 persistence not met          → NORMAL
      Layer 1 met, ML not confirmed         → ADVISORY  (watch mode)
      Both layers confirmed                 → ANOMALY_CONFIRMED
    """
    layer1 = RuleEngine().check_batch(batch_df)

    # Entire batch was in defrost — nothing to evaluate
    if layer1.get("suppressed") and layer1.get("reason") == "ENTIRE_BATCH_IN_DEFROST":
        return {
            "decision":  "NORMAL",
            "severity":  "NORMAL",
            "confirmed": False,
            "reason":    "Entire batch suppressed for defrost",
            "layer1":    layer1,
            "layer2":    {
                "confirmed":    False,
                "worst_score":  0.0,
                "anomaly_pct":  0.0,
            },
        }

    layer2_detector = IsolationForestDetector(model_dir=model_dir)
    layer2          = layer2_detector.predict(batch_df, previous_tail=previous_tail)

    persist = layer1["persistence_confirmed"]
    ml_conf = layer2["confirmed"]

    if not persist:
        decision  = "NORMAL"
        severity  = "NORMAL"
        confirmed = False
        reason    = "Layer 1 persistence not met"
    elif persist and not ml_conf:
        decision  = "ADVISORY"
        severity  = "ADVISORY"
        confirmed = False
        reason    = "Layer 1 flagged but ML not confirmed"
    else:
        decision  = "ANOMALY_CONFIRMED"
        severity  = layer1["worst_severity"]
        confirmed = True
        reason    = f"Both layers confirmed. Max z={layer1['max_z_score']}"

    return {
        "decision":  decision,
        "severity":  severity,
        "confirmed": confirmed,
        "reason":    reason,
        "layer1":    layer1,
        "layer2":    layer2,
    }


# ─────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ANOMALY DETECTOR — Self-test (Defrost Suppression)")
    print("=" * 60)

    import numpy as np

    # ── Test 1: batch with both defrost columns present ───────────────────────
    df_with_defrost = pd.DataFrame({
        "suction_pressure_bar": [4.2] * 10,
        "is_defrost_cycle":     [1] * 5 + [0] * 5,
        "defrost_recovery":     [0] * 10,
    })
    res1 = RuleEngine().check_batch(df_with_defrost)
    assert res1["suppressed_rows"] == 5, "Expected 5 defrost rows suppressed"
    print(f"Test 1 (defrost columns present): suppressed_rows={res1['suppressed_rows']} ✓")

    # ── Test 2: batch WITHOUT defrost columns (external data source) ──────────
    df_no_defrost = pd.DataFrame({
        "suction_pressure_bar": [4.2] * 10,
    })
    res2 = RuleEngine().check_batch(df_no_defrost)
    assert res2["suppressed_rows"] == 0, "Expected 0 suppressed rows when columns absent"
    print(f"Test 2 (no defrost columns):      suppressed_rows={res2['suppressed_rows']} ✓")

    # ── Test 3: batch with is_defrost_cycle but NO defrost_recovery ───────────
    df_partial = pd.DataFrame({
        "suction_pressure_bar": [4.2] * 10,
        "is_defrost_cycle":     [0] * 8 + [1] * 2,
        # defrost_recovery column intentionally absent
    })
    res3 = RuleEngine().check_batch(df_partial)
    assert res3["suppressed_rows"] == 2, "Expected 2 defrost rows, 0 recovery rows"
    print(f"Test 3 (partial defrost columns): suppressed_rows={res3['suppressed_rows']} ✓")

    # ── Test 4: entire batch in defrost ───────────────────────────────────────
    df_all_defrost = pd.DataFrame({
        "suction_pressure_bar": [4.2] * 10,
        "is_defrost_cycle":     [1] * 10,
        "defrost_recovery":     [0] * 10,
    })
    res4 = RuleEngine().check_batch(df_all_defrost)
    assert res4.get("reason") == "ENTIRE_BATCH_IN_DEFROST"
    print(f"Test 4 (all defrost):             reason={res4['reason']} ✓")

    print("\n✓ anomaly_detector.py — OK")