"""
sensor_health.py
═══════════════════════════════════════════════════════
Sensor integrity monitoring for HVAC systems.
Detects stuck values, zero/null readings, and physical impossibility.

Business Logic:
  - STUCK: 10 mins of identical readings (physically impossible)
  - ZERO_OR_NULL: Disconnected or dead sensor
  - IMPOSSIBLE: Physics-based hard limits (e.g. suction > 8 bar)
  - CRITICAL: If these sensors fail, we cannot trust ML results.

FIX (sensor fault recording):
  Previously, fault_type / rows_affected / sample_value were shared
  mutable variables across the three checks (STUCK, ZERO_OR_NULL,
  IMPOSSIBLE).  If a sensor triggered only the first check, the
  variables were stale or unbound when the final recording block ran,
  producing wrong fault_type labels and occasional NameErrors.

  Each check now records its own fault entry immediately and
  independently.  The sensor is added to faulty_sensors on the
  first fault found (earliest check wins — most actionable cause
  reported first).  Multiple fault types on the same sensor are
  all stored in fault_details[sensor]["all_faults"] for traceability.
═══════════════════════════════════════════════════════
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

# Hard physical boundaries — impossible to exceed regardless of fault
PHYSICAL_LIMITS = {
    "suction_pressure_bar":    {"min": 0.5,  "max": 8.0},
    "discharge_pressure_bar":  {"min": 5.0,  "max": 20.0},
    "coil_delta_T_C":          {"min": -5.0, "max": 25.0},
    "superheat_C":             {"min": -2.0, "max": 30.0},
    "subcooling_C":            {"min": 0.0,  "max": 20.0},
    "chilled_water_flow_LPS":  {"min": 5.0,  "max": 120.0},
    "chilled_water_delta_T_C": {"min": 0.5,  "max": 15.0},
    "blower_current_A":        {"min": 2.0,  "max": 25.0},
    "compressor_current_A":    {"min": 5.0,  "max": 80.0},
    "supply_air_temp_C":       {"min": 5.0,  "max": 35.0},
    "return_air_temp_C":       {"min": 10.0, "max": 45.0},
    "ambient_temp_C":          {"min": -5.0, "max": 55.0},
    "compressor_runtime_pct":  {"min": 0.0,  "max": 100.0},
}

# If these fail → skip anomaly detection entirely (garbage in, garbage out)
CRITICAL_SENSORS = [
    "suction_pressure_bar",
    "discharge_pressure_bar",
    "coil_delta_T_C",
    "compressor_current_A",
]


class SensorHealthChecker:
    """
    Validates sensor data integrity before it reaches the ML pipeline.
    """

    def check_batch(self, df: pd.DataFrame) -> dict:
        """
        Run 3 integrity checks per sensor: STUCK, ZERO_OR_NULL, IMPOSSIBLE.

        Each check records its findings independently.  The primary
        fault_type reported is from the first check that fires (STUCK →
        ZERO_OR_NULL → IMPOSSIBLE), which is the most operationally
        actionable cause.  All faults found on a sensor are stored in
        fault_details[sensor]["all_faults"] for full traceability.
        """
        faulty_sensors = []
        fault_details  = {}

        for sensor, limits in PHYSICAL_LIMITS.items():
            if sensor not in df.columns:
                continue

            series     = df[sensor]
            sensor_faults: list[dict] = []   # collects all faults for this sensor

            # ── CHECK 1 — Stuck value (10-row rolling window) ────────────────
            # 10 consecutive identical readings are physically impossible for
            # any live HVAC sensor; indicates a frozen/dead transmitter.
            if len(series) >= 10:
                rolling_std = series.rolling(window=10).std()
                stuck_mask  = rolling_std < 0.001
                if stuck_mask.any():
                    sensor_faults.append({
                        "fault_type":    "STUCK",
                        "rows_affected": int(stuck_mask.sum()),
                        "sample_value":  float(series[stuck_mask].iloc[0]),
                    })

            # ── CHECK 2 — Zero or null ────────────────────────────────────────
            # NaN or 0.0 on a pressure/temperature/flow sensor almost always
            # means a disconnected wire, blown fuse, or dead transmitter.
            null_mask = series.isna() | (series == 0.0)
            if null_mask.any():
                sensor_faults.append({
                    "fault_type":    "ZERO_OR_NULL",
                    "rows_affected": int(null_mask.sum()),
                    "sample_value":  0.0,
                })

            # ── CHECK 3 — Physically impossible value ─────────────────────────
            imp_mask = (series < limits["min"]) | (series > limits["max"])
            if imp_mask.any():
                sensor_faults.append({
                    "fault_type":    "IMPOSSIBLE_VALUE",
                    "rows_affected": int(imp_mask.sum()),
                    "sample_value":  float(series[imp_mask].iloc[0]),
                })

            # ── Record if any check fired ─────────────────────────────────────
            if sensor_faults:
                faulty_sensors.append(sensor)
                primary = sensor_faults[0]          # first fault = primary report
                fault_details[sensor] = {
                    "fault_type":    primary["fault_type"],      # primary (most actionable)
                    "rows_affected": primary["rows_affected"],
                    "sample_value":  primary["sample_value"],
                    "all_faults":    sensor_faults,              # full audit trail
                }

        has_fault      = len(faulty_sensors) > 0
        critical_fault = any(s in CRITICAL_SENSORS for s in faulty_sensors)

        if critical_fault:
            recommendation = "SKIP_ANOMALY_DETECTION"
        elif has_fault:
            recommendation = "PROCEED_WITH_CAUTION"
        else:
            recommendation = "PROCEED"

        return {
            "has_fault":       has_fault,
            "critical_fault":  critical_fault,
            "faulty_sensors":  faulty_sensors,
            "fault_details":   fault_details,
            "recommendation":  recommendation,
            "alert_type":      "SENSOR_FAULT" if has_fault else None,
        }

    def generate_sensor_fault_ticket(
        self,
        unit_id: str,
        fault_details: dict,
    ) -> Optional[dict]:
        """
        Generate or update a ticket when sensor health check fails.
        """
        from ticket_engine import (
            IncidentManager, STATUS_OPEN, SEVERITY_MAP,
            TICKETS_DIR, COOLDOWN_HOURS,
        )

        incident_type  = "SENSOR_FAULT"
        mgr            = IncidentManager()
        critical_fault = any(s in CRITICAL_SENSORS for s in fault_details.keys())
        severity       = "CRITICAL" if critical_fault else "WARNING"

        # Cooldown — suppress duplicate sensor tickets
        if mgr.check_cooldown(unit_id, incident_type, severity):
            print(f"[TICKET SUPPRESSED - DUPLICATE] {unit_id} {incident_type} in cooldown")
            return None

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ticket_id = f"SENSOR-{unit_id}-{timestamp}"

        repair_cost = 12_000 if critical_fault else 5_000

        ticket = {
            "ticket_id":      ticket_id,
            "incident_type":  incident_type,
            "unit_id":        unit_id,
            "generated_at":   datetime.now().isoformat(),
            "status":         STATUS_OPEN,
            "severity":       severity,
            "mapped_severity": SEVERITY_MAP.get(severity, "LOW"),
            "urgency": {
                "hours":        24 if critical_fault else 72,
                "display":      "24 hours \u2014 URGENT" if critical_fault else "3 days",
                "urgency_band": "EMERGENCY" if critical_fault else "CRITICAL",
                "message": (
                    "Sensor fault blinds anomaly detection. "
                    "Repair required immediately to restore system visibility."
                ),
            },
            "cost_escalation": {
                "daily_energy_excess_inr":          0,
                "weekly_energy_excess_inr":         0,
                "repair_cost_now_inr":              repair_cost,
                "repair_cost_if_ignored_1week_inr": int(repair_cost * 1.8),
                "repair_cost_if_ignored_2weeks_inr": int(repair_cost * 3.2),
                "repair_cost_if_ignored_4weeks_inr": int(repair_cost * 5.6),
                "emergency_cost_inr":               74_000,
                "cost_per_day_delay_inr":           repair_cost * 0.1,
            },
            "faulty_sensors":  list(fault_details.keys()),
            "fault_details":   fault_details,
            "sensor_snapshot": {s: d["sample_value"] for s, d in fault_details.items()},
            "action_plan": [
                f"Locate sensor on unit {unit_id}",
                "Inspect wiring and connector at sensor head",
                "Check sensor power supply voltage (should be 24 VDC)",
                "If wiring intact: replace sensor module",
                "After replacement: verify reading is within normal range",
                "Recalibrate against reference instrument",
                "Resume normal HVAC anomaly monitoring",
            ],
            "note": (
                "Anomaly detection was SKIPPED for this batch due to sensor fault. "
                "HVAC health unknown until sensor is restored."
            ),
        }

        final_ticket = mgr.update_or_create(unit_id, incident_type, ticket)

        TICKETS_DIR.mkdir(parents=True, exist_ok=True)
        with open(TICKETS_DIR / f"{final_ticket['ticket_id']}.json", "w") as f:
            json.dump(final_ticket, f, indent=2)

        return final_ticket


# ─────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SENSOR HEALTH CHECKER — Self-test")
    print("=" * 60)

    import numpy as np

    # Build a test frame that exercises all three fault types
    rng  = np.random.default_rng(0)
    data = {
        # STUCK: suction pressure repeats the same value for 12 rows
        "suction_pressure_bar":    [4.2] * 12 + list(rng.normal(4.2, 0.15, 3)),
        # ZERO_OR_NULL: two null readings mid-batch
        "discharge_pressure_bar":  [11.0] * 7 + [None, None] + [11.0] * 6,
        # IMPOSSIBLE: one reading above physical max (25°C for coil_delta_T)
        "coil_delta_T_C":          [8.0] * 14 + [99.0],
        # IMPOSSIBLE + STUCK combo on compressor current
        "compressor_current_A":    [38.0] * 15,
        # Healthy sensor — should not appear in fault_details
        "ambient_temp_C":          list(rng.normal(28.0, 2.0, 15)),
    }
    df = pd.DataFrame(data)

    checker = SensorHealthChecker()
    result  = checker.check_batch(df)

    print(f"\nHas Fault:       {result['has_fault']}")
    print(f"Critical Fault:  {result['critical_fault']}")
    print(f"Recommendation:  {result['recommendation']}")
    print(f"Faulty Sensors:  {result['faulty_sensors']}")
    print("\nFault Details:")
    for sensor, detail in result["fault_details"].items():
        print(f"  {sensor}:")
        print(f"    Primary fault : {detail['fault_type']}")
        print(f"    Rows affected : {detail['rows_affected']}")
        print(f"    Sample value  : {detail['sample_value']}")
        all_types = [f["fault_type"] for f in detail["all_faults"]]
        print(f"    All faults    : {all_types}")

    print("\n✓ sensor_health.py — OK")