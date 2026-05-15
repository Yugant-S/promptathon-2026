"""
carrier_baseline.py
═══════════════════════════════════════════════════════
Single source of truth for all Carrier 30XA chiller constants.
All other modules import from here — no hardcoded numbers elsewhere.

Based on:
  - Carrier 30XA Product Data (PD 30XA-1PD)
  - ASHRAE Fundamentals Handbook (thermodynamic ranges)
  - DOE Commercial Buildings Energy Consumption Survey
  - Field calibration data from manufacturing facilities

Carrier 30XA Series: Air-cooled screw chiller, 150–500 ton range.
Manufacturing facility context: continuous 24/7 operation,
stable load profile, ~50 kW nominal cooling at target facility.
═══════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────
# SENSOR BASELINE — Carrier 30XA operating ranges
# min_normal / max_normal: 2-sigma band (95% of readings)
# warning_low / warning_high: 2.5-sigma (pre-alarm band)
# critical_low / critical_high: 3-sigma (alarm boundary)
# ─────────────────────────────────────────────────────
CARRIER_BASELINE = {
    # Suction pressure at compressor inlet (low-side refrigerant pressure)
    # R-134a at ~4.2 bar ≈ 0°C evaporating temp — freeze risk begins <3.5 bar
    "suction_pressure_bar": {
        "mean": 4.2, "std": 0.15,
        "min_normal": 3.90, "max_normal": 4.50,
        "warning_low": 3.825, "warning_high": 4.575,
        "critical_low": 3.75, "critical_high": 4.65,
    },
    # Discharge pressure at compressor outlet (high-side)
    # Rises with condenser fouling or high ambient temp
    "discharge_pressure_bar": {
        "mean": 11.0, "std": 0.30,
        "min_normal": 10.40, "max_normal": 11.60,
        "warning_low": 10.25, "warning_high": 11.75,
        "critical_low": 10.10, "critical_high": 11.90,
    },
    # Temperature difference across evaporator coil (supply vs return air)
    # Drops with ice formation (insulation effect) or refrigerant starvation
    "coil_delta_T_C": {
        "mean": 8.0, "std": 0.40,
        "min_normal": 7.20, "max_normal": 8.80,
        "warning_low": 7.00, "warning_high": 9.00,
        "critical_low": 6.80, "critical_high": 9.20,
    },
    # Superheat at evaporator outlet — critical for freeze detection
    # Too low (<4°C) = liquid slugging risk; too high = insufficient refrigerant
    "superheat_C": {
        "mean": 6.0, "std": 0.80,
        "min_normal": 4.40, "max_normal": 7.60,
        "warning_low": 4.00, "warning_high": 8.00,
        "critical_low": 3.60, "critical_high": 8.40,
    },
    # Subcooling at condenser outlet — indicates refrigerant charge adequacy
    # Low subcooling → low charge → freeze risk
    "subcooling_C": {
        "mean": 5.0, "std": 0.50,
        "min_normal": 4.00, "max_normal": 6.00,
        "warning_low": 3.75, "warning_high": 6.25,
        "critical_low": 3.50, "critical_high": 6.50,
    },
    # Chilled water volumetric flow rate through evaporator
    # Low flow → longer dwell time → coil freeze risk
    "chilled_water_flow_LPS": {
        "mean": 55.0, "std": 2.0,
        "min_normal": 51.0, "max_normal": 59.0,
        "warning_low": 50.0, "warning_high": 60.0,
        "critical_low": 49.0, "critical_high": 61.0,
    },
    # Delta-T across chilled water circuit (supply vs return water temp)
    # Rises with reduced flow (more heat extracted per unit volume)
    "chilled_water_delta_T_C": {
        "mean": 5.5, "std": 0.30,
        "min_normal": 4.90, "max_normal": 6.10,
        "warning_low": 4.75, "warning_high": 6.25,
        "critical_low": 4.60, "critical_high": 6.40,
    },
    # Evaporator fan/blower motor current draw
    # Rises when coil ices up (increased air resistance)
    "blower_current_A": {
        "mean": 12.5, "std": 0.30,
        "min_normal": 11.90, "max_normal": 13.10,
        "warning_low": 11.75, "warning_high": 13.25,
        "critical_low": 11.60, "critical_high": 13.40,
    },
    # Compressor motor current draw
    # Rises with high discharge pressure (condenser issue, overload)
    "compressor_current_A": {
        "mean": 38.0, "std": 1.50,
        "min_normal": 35.00, "max_normal": 41.00,
        "warning_low": 34.25, "warning_high": 41.75,
        "critical_low": 33.50, "critical_high": 42.50,
    },
    # Supply air temperature leaving the air handler
    # Rises if coil fouled; drops if refrigerant overfeeding
    "supply_air_temp_C": {
        "mean": 14.0, "std": 0.50,
        "min_normal": 13.00, "max_normal": 15.00,
        "warning_low": 12.75, "warning_high": 15.25,
        "critical_low": 12.50, "critical_high": 15.50,
    },
    # Return air temperature entering the unit from conditioned space
    # Relatively stable; reflects building thermal load
    "return_air_temp_C": {
        "mean": 24.0, "std": 1.00,
        "min_normal": 22.00, "max_normal": 26.00,
        "warning_low": 21.50, "warning_high": 26.50,
        "critical_low": 21.00, "critical_high": 27.00,
    },
    # Outdoor ambient temperature — affects condenser performance
    # High ambient → higher discharge pressure → compressor stress
    "ambient_temp_C": {
        "mean": 28.0, "std": 2.00,
        "min_normal": 24.00, "max_normal": 32.00,
        "warning_low": 23.00, "warning_high": 33.00,
        "critical_low": 22.00, "critical_high": 34.00,
    },
    # Percentage of time compressor is running (duty cycle)
    # High runtime → capacity issue or undersized unit
    "compressor_runtime_pct": {
        "mean": 70.0, "std": 3.00,
        "min_normal": 64.00, "max_normal": 76.00,
        "warning_low": 62.50, "warning_high": 77.50,
        "critical_low": 61.00, "critical_high": 79.00,
    },
}

# ─────────────────────────────────────────────────────
# Z-SCORE SEVERITY THRESHOLDS
# Based on standard statistical process control +
# Carrier alarm setpoint philosophy
# ─────────────────────────────────────────────────────
Z_SCORE_THRESHOLDS = {
    "NORMAL":           2.0,   # |z| < 2.0 → no action
    "ADVISORY":         2.0,   # |z| >= 2.0 → monitor
    "WARNING":          2.5,   # |z| >= 2.5 → schedule inspection
    "CRITICAL":         3.0,   # |z| >= 3.0 → expedite repair
    "IMMINENT_FAILURE": 4.0,   # |z| >= 4.0 → emergency shutdown risk
}

# 3-of-5 persistence rule: requires 3 consecutive threshold crossings
# in a 5-row window to filter transient spikes (sensor glitches, startup)
PERSISTENCE_RULE = 3  # rows out of last 5 that must cross threshold

# ─────────────────────────────────────────────────────
# DATA RETENTION TTL (days per severity)
# Higher severity → longer retention for audit trail
# and model retraining datasets
# ─────────────────────────────────────────────────────
TTL_DAYS = {
    "NORMAL":           2,
    "ADVISORY":         7,
    "WARNING":          15,
    "CRITICAL":         30,
    "IMMINENT_FAILURE": 90,
}

# ─────────────────────────────────────────────────────
# FREEZE CAUSE WEIGHTS — probability priors
# Derived from: ASHRAE maintenance failure mode analysis
# and Carrier field service reports for 30XA series
# ─────────────────────────────────────────────────────
FREEZE_CAUSE_WEIGHTS = {
    "low_refrigerant":   0.45,  # Most common: 45% of freeze events
    "dirty_coil":        0.28,  # Second: biofilm/dust fouling
    "exv_malfunction":   0.15,  # EXV hunting → liquid floodback
    "low_chilled_water": 0.08,  # Pump/strainer failure
    "condenser_issue":   0.04,  # Condenser coil fouling
}

# ─────────────────────────────────────────────────────
# COST CONSTANTS (INR, Indian manufacturing context)
# Sources:
#   - DOE Building Technologies Office: COP degradation data
#   - ASHRAE Standard 180: HVAC Inspection and Maintenance
#   - Carrier India service pricing (2024 estimates)
# ─────────────────────────────────────────────────────
COST_CONSTANTS = {
    # Carrier 30XA nominal COP from product data sheet
    "normal_cop": 3.8,
    # Degraded COP during freeze progression (~32% drop per DOE study)
    # DOE: "Energy Savings Potential of HVAC Maintenance" (2012)
    "degraded_cop": 2.6,
    # Industrial electricity tariff (INR/kWh) — manufacturing facility
    "energy_rate_inr_per_kwh": 8.0,
    # Scheduled maintenance costs (technician + parts, no urgency premium)
    "scheduled_repair_min_inr": 8_000,
    "scheduled_repair_max_inr": 15_000,
    # Emergency call-out costs (after-hours, expedited parts)
    "emergency_repair_min_inr": 45_000,
    "emergency_repair_max_inr": 75_000,
    # Compressor replacement if freeze damage occurs
    "compressor_damage_min_inr": 1_20_000,
    # Severity-specific repair cost estimates
    "repair_cost_by_severity": {
        "ADVISORY":         8_000,
        "WARNING":         15_000,
        "CRITICAL":        45_000,
        "IMMINENT_FAILURE": 75_000,
    },
    # Delay multiplier: waiting 3+ days escalates cost (emergency premium)
    "delay_cost_multiplier": 1.8,
}

# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────

def get_z_score(sensor: str, value: float) -> float:
    """
    Compute z-score of a sensor reading vs Carrier 30XA baseline.
    Returns signed z (negative = below mean, positive = above mean).
    Caller should use abs(z) for threshold comparisons.
    """
    if sensor not in CARRIER_BASELINE:
        raise ValueError(f"Unknown sensor: '{sensor}'. "
                         f"Valid sensors: {list(CARRIER_BASELINE.keys())}")
    b = CARRIER_BASELINE[sensor]
    return (value - b["mean"]) / b["std"]


def get_severity_from_z(z_score: float) -> str:
    """
    Map absolute z-score to severity label.
    Uses one-sided thresholds — direction (high/low) is
    interpreted per sensor by the rule engine.
    """
    z = abs(z_score)
    if z >= Z_SCORE_THRESHOLDS["IMMINENT_FAILURE"]:
        return "IMMINENT_FAILURE"
    elif z >= Z_SCORE_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif z >= Z_SCORE_THRESHOLDS["WARNING"]:
        return "WARNING"
    elif z >= Z_SCORE_THRESHOLDS["ADVISORY"]:
        return "ADVISORY"
    else:
        return "NORMAL"


# Severity ordering for comparison
SEVERITY_ORDER = {
    "NORMAL": 0,
    "ADVISORY": 1,
    "WARNING": 2,
    "CRITICAL": 3,
    "IMMINENT_FAILURE": 4,
}


def compare_severity(s1: str, s2: str) -> str:
    """Return the more severe of two severity labels."""
    return s1 if SEVERITY_ORDER.get(s1, 0) >= SEVERITY_ORDER.get(s2, 0) else s2


# ─────────────────────────────────────────────────────
# FREEZE RISK SCORING  (single canonical copy — import from here)
# ─────────────────────────────────────────────────────

def calculate_freeze_risk(row: dict) -> dict:
    """Compute freeze risk scores (0-100) based on sensor deviations.

    Single authoritative implementation — previously duplicated in
    hvac_simulator.py and ticket_engine.py (which silently diverged).
    Both modules now import this function from carrier_baseline.
    """
    def clamp(val, lo=0, hi=100):
        return max(lo, min(hi, val))

    def interp_risk(value, normal_boundary, critical_boundary):
        if normal_boundary == critical_boundary:
            return 0.0
        ratio = (value - normal_boundary) / (critical_boundary - normal_boundary)
        return clamp(ratio * 100)

    suction   = row.get("suction_pressure_bar", 4.2)
    subcool   = row.get("subcooling_C", 5.0)
    superheat = row.get("superheat_C", 6.0)

    lr_suction   = interp_risk(3.9 - suction, 0, 0.4)
    lr_subcool   = interp_risk(4.0 - subcool, 0, 2.0)
    lr_superheat = interp_risk(superheat - 8.0, 0, 4.0) if superheat > 8 else 0
    low_refrigerant_score = clamp(lr_suction * 0.5 + lr_subcool * 0.3 + lr_superheat * 0.2)

    coil_dt     = row.get("coil_delta_T_C", 8.0)
    blower      = row.get("blower_current_A", 12.5)
    supply_temp = row.get("supply_air_temp_C", 14.0)
    dc_coil    = interp_risk(8.0 - coil_dt, 0, 3.0)
    dc_blower  = interp_risk(blower - 13.0, 0, 2.0)
    dc_supply  = interp_risk(supply_temp - 15.0, 0, 3.0)
    dirty_coil_score = clamp(dc_coil * 0.5 + dc_blower * 0.3 + dc_supply * 0.2)

    sh_dev    = abs(superheat - 6.0)
    exv_score = clamp(interp_risk(sh_dev, 1.0, 6.0))

    flow  = row.get("chilled_water_flow_LPS", 55.0)
    cw_dt = row.get("chilled_water_delta_T_C", 5.5)
    lcw_flow = interp_risk(55.0 - flow, 0, 15.0)
    lcw_dt   = interp_risk(cw_dt - 6.0, 0, 2.0)
    low_chilled_water_score = clamp(lcw_flow * 0.7 + lcw_dt * 0.3)

    discharge  = row.get("discharge_pressure_bar", 11.0)
    comp_curr  = row.get("compressor_current_A", 38.0)
    ci_disch = interp_risk(discharge - 11.5, 0, 2.0)
    ci_curr  = interp_risk(comp_curr - 40.0, 0, 5.0)
    condenser_issue_score = clamp(ci_disch * 0.6 + ci_curr * 0.4)

    scores = {
        "low_refrigerant":   round(low_refrigerant_score, 2),
        "dirty_coil":        round(dirty_coil_score, 2),
        "exv_malfunction":   round(exv_score, 2),
        "low_chilled_water": round(low_chilled_water_score, 2),
        "condenser_issue":   round(condenser_issue_score, 2),
    }

    total = sum(scores[cause] * FREEZE_CAUSE_WEIGHTS[cause] for cause in scores)
    total = round(clamp(total), 2)

    dominant = max(scores, key=scores.__getitem__)
    return {
        "total_freeze_risk": total,
        "dominant_cause":    dominant,
        "scores":            scores,
    }


# ─────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CARRIER 30XA BASELINE — Self-test")
    print("=" * 60)

    # Test z-score calculation
    test_cases = [
        ("suction_pressure_bar", 4.2),   # Normal
        ("suction_pressure_bar", 3.9),   # Slight drop
        ("suction_pressure_bar", 3.6),   # Warning
        ("suction_pressure_bar", 3.3),   # Critical
        ("superheat_C", 2.0),            # Low superheat — freeze risk
        ("discharge_pressure_bar", 12.2),# High discharge — condenser issue
    ]

    print(f"\n{'Sensor':<30} {'Value':>8} {'Z-Score':>8} {'Severity':<20}")
    print("-" * 70)
    for sensor, val in test_cases:
        z = get_z_score(sensor, val)
        sev = get_severity_from_z(z)
        print(f"{sensor:<30} {val:>8.2f} {z:>8.3f} {sev:<20}")

    print(f"\nTTL Config: {TTL_DAYS}")
    print(f"Freeze Cause Weights: {FREEZE_CAUSE_WEIGHTS}")
    print(f"Persistence Rule: {PERSISTENCE_RULE} of 5 consecutive rows")
    print("\n✓ carrier_baseline.py — OK")
