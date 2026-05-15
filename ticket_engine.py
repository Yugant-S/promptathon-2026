"""
ticket_engine.py

Generates structured maintenance tickets from confirmed anomalies.
Upgraded with enterprise-grade incident management and predictive urgency timelines.

Ticket lifecycle:
  OPEN -> IN_PROGRESS -> RESOLVED -> CLOSED

Cost model sources:
  - ASHRAE Standard 180-2012: HVAC Inspection and Maintenance
  - DOE Building Technologies Office: "Energy Savings Potential
    of HVAC Equipment Maintenance" (ORNL/TM-2004/158)
  - Carrier India field service pricing (2024)
  - Indian industrial electricity tariff: Rs 8/kWh

FIX — generate_ticket KeyError on cause scores:
  calculate_freeze_risk() returns:
      {
          "total_freeze_risk": float,
          "dominant_cause":    str,
          "scores":            {"low_refrigerant": float, ...}
      }
  Previously, generate_ticket built avg_risk by iterating over
  FREEZE_CAUSE_WEIGHTS keys and doing r[cause] on the top-level
  result dict — those keys don't exist there, they live in r["scores"].
  Fixed to extract r["scores"] first, then average across rows.
  avg_total extraction (r["total_freeze_risk"]) was already correct
  and is unchanged.
"""

import json
import uuid
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from carrier_baseline import (
    CARRIER_BASELINE, FREEZE_CAUSE_WEIGHTS, COST_CONSTANTS,
    get_z_score, get_severity_from_z, SEVERITY_ORDER, compare_severity,
    calculate_freeze_risk,
)

TICKETS_DIR    = Path("alerts/tickets")
REGISTRY_PATH  = Path("data/incident_registry.json")

# ─────────────────────────────────────────────────────
# LIFECYCLE STATES
# ─────────────────────────────────────────────────────
STATUS_OPEN        = "OPEN"
STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_RESOLVED    = "RESOLVED"
STATUS_CLOSED      = "CLOSED"

# ─────────────────────────────────────────────────────
# FILTERING & COOLDOWN
# ─────────────────────────────────────────────────────
MIN_CONFIDENCE_PCT = 30.0   # Min anomaly_pct from Layer 2
COOLDOWN_HOURS     = 2
RECOVERY_MINUTES   = 120    # 2 hours of normal data to auto-resolve

# ─────────────────────────────────────────────────────
# SEVERITY MAPPING
# ─────────────────────────────────────────────────────
SEVERITY_MAP = {
    "ADVISORY":         "LOW",
    "WARNING":          "MEDIUM",
    "CRITICAL":         "HIGH",
    "IMMINENT_FAILURE": "CRITICAL",
}
REVERSE_SEVERITY_MAP = {v: k for k, v in SEVERITY_MAP.items()}
MIN_TICKET_SEVERITY  = "MEDIUM"


# ─────────────────────────────────────────────────────
# DOMINANT CAUSE HELPER
# ─────────────────────────────────────────────────────

def get_dominant_cause(avg_scores: dict) -> str:
    """
    Return the cause with the highest weighted contribution.

    Parameters
    ----------
    avg_scores : dict
        Flat dict of {cause_name: avg_score} — already extracted from
        calculate_freeze_risk()["scores"] and averaged across batch rows.
    """
    causes = list(FREEZE_CAUSE_WEIGHTS.keys())
    return max(causes, key=lambda c: avg_scores.get(c, 0) * FREEZE_CAUSE_WEIGHTS[c])


# ─────────────────────────────────────────────────────
# URGENCY ESTIMATOR
# ─────────────────────────────────────────────────────

def estimate_urgency(severity: str, freeze_risk_pct: float) -> dict:
    """
    Estimate time remaining before hard failure.
    Returns a structured urgency object with human-readable display.
    """
    risk = freeze_risk_pct
    days = 28   # conservative default

    if severity == "ADVISORY":
        days = 28 if risk < 30 else 21
    elif severity == "WARNING":
        days = 14 if risk < 52 else 7
    elif severity == "CRITICAL":
        days = 5  if risk < 75 else 2
    elif severity == "IMMINENT_FAILURE":
        days = 2  if risk < 92 else 0.25   # 6 hours

    hours = int(days * 24)

    if days >= 28:  band = "LOW"
    elif days >= 14: band = "MEDIUM"
    elif days >= 7:  band = "HIGH"
    elif days >= 1:  band = "CRITICAL"
    else:            band = "EMERGENCY"

    if days >= 14:
        display = f"~{round(days / 7)} weeks"
    elif days >= 1:
        display = f"{int(days)} days"
    else:
        display = f"{int(hours)} hours \u2014 URGENT"

    messages = {
        "LOW":       ("Early degradation signature detected. Unit has approximately "
                      "4 weeks before performance impact. Schedule routine maintenance "
                      "at next convenient window."),
        "MEDIUM":    ("Developing anomaly confirmed. Schedule maintenance within "
                      "2\u20133 weeks to prevent escalation to critical failure."),
        "HIGH":      ("Significant degradation. Maintenance required this week. "
                      "Cost escalating daily \u2014 see cost breakdown below."),
        "CRITICAL":  ("Failure approaching within days. Dispatch technician "
                      "immediately. Production risk if unaddressed."),
        "EMERGENCY": ("Imminent failure. Emergency dispatch required. "
                      "Unit may trip within hours."),
    }

    return {
        "hours":        hours,
        "display":      display,
        "urgency_band": band,
        "message":      messages.get(band, ""),
    }


# ─────────────────────────────────────────────────────
# ACTION PLAN GENERATOR
# ─────────────────────────────────────────────────────

_ACTION_PLANS = {
    "low_refrigerant": [
        "Connect manifold gauge set; verify suction pressure against P-T chart.",
        "Inspect all refrigerant circuit joints and service valves for leaks.",
        "Use electronic leak detector; confirm with UV dye if necessary.",
        "Recover charge, evacuate to <500 microns, weigh-in factory charge.",
        "Log refrigerant quantity; schedule 30-day follow-up check.",
    ],
    "dirty_coil": [
        "Perform LOTO; inspect coil fins for biofilm and debris.",
        "Apply Carrier-approved coil cleaner; dwell 10 min. Rinse thoroughly.",
        "Replace air filters (MERV-8 minimum); inspect blower wheel.",
        "Restore power; verify coil delta_T recovers within 30 minutes.",
        "Schedule next coil inspection for 90 days out.",
    ],
    "exv_malfunction": [
        "Check EXV wiring harness for loose connections or corrosion.",
        "Measure superheat at evaporator outlet; target 6 \u00b1 2\u00b0C.",
        "Run EXV calibration routine via controller service menu.",
        "If oscillation persists, replace EXV stepper motor assembly.",
        "Monitor superheat for 2 hours post-repair for stability.",
    ],
    "low_chilled_water": [
        "Inspect pump inlet strainer; clean if >50% blocked.",
        "Check VFD frequency and current vs design flow rate.",
        "Verify balancing valve positions in chilled water loop.",
        "Adjust balancing valves if flow is >10% below design.",
        "Inspect pump impeller for wear if flow remains low.",
    ],
    "condenser_issue": [
        "Inspect condenser fins for fouling or physical damage.",
        "Verify all condenser fans are rotating in correct direction.",
        "Check head pressure vs ambient temperature correlation.",
        "Clean condenser coil with low-pressure water (max 2 bar).",
        "Monitor head pressure trend over next 24 hours.",
    ],
}


def get_action_plan(dominant_cause: str, severity: str) -> list:
    plan = _ACTION_PLANS.get(dominant_cause, _ACTION_PLANS["condenser_issue"])
    if severity == "IMMINENT_FAILURE":
        plan = ["IMMEDIATE: Consider unit shutdown to prevent damage."] + plan[:4]
    return plan[:5]


# ─────────────────────────────────────────────────────
# COST CALCULATOR
# ─────────────────────────────────────────────────────

def calculate_costs(
    severity: str,
    freeze_risk_pct: float,
    unit_kw: float = 50.0,
) -> dict:
    """
    Estimate energy waste and repair cost escalation over a 4-week horizon.

    Energy waste is computed from the COP degradation that accompanies
    freeze-risk progression (DOE ORNL/TM-2004/158).
    """
    cc          = COST_CONSTANTS
    rate        = cc["energy_rate_inr_per_kwh"]
    risk_factor = freeze_risk_pct / 100.0

    effective_cop      = cc["normal_cop"] - (cc["normal_cop"] - cc["degraded_cop"]) * risk_factor
    energy_waste_kw_day = unit_kw * (1.0 / effective_cop - 1.0 / cc["normal_cop"]) * 24
    daily_energy_cost   = energy_waste_kw_day * rate

    repair_now = cc["repair_cost_by_severity"].get(severity, 8_000)
    repair_1w  = repair_now * 1.8
    repair_2w  = repair_now * 3.2
    repair_4w  = repair_now * 5.6
    emergency  = 74_000

    cost_per_day_delay = daily_energy_cost + (repair_1w - repair_now) / 7.0

    return {
        "daily_energy_excess_inr":           round(daily_energy_cost, 2),
        "weekly_energy_excess_inr":          round(daily_energy_cost * 7, 2),
        "repair_cost_now_inr":               int(repair_now),
        "repair_cost_if_ignored_1week_inr":  int(repair_1w),
        "repair_cost_if_ignored_2weeks_inr": int(repair_2w),
        "repair_cost_if_ignored_4weeks_inr": int(repair_4w),
        "emergency_cost_inr":                emergency,
        "cost_per_day_delay_inr":            round(cost_per_day_delay, 2),
        "source": "ASHRAE Std 180 + DOE ORNL/TM-2004/158 + Carrier India field data",
    }


# ─────────────────────────────────────────────────────
# INCIDENT MANAGER
# ─────────────────────────────────────────────────────

class IncidentManager:
    """
    Enterprise incident lifecycle: deduplication, escalation, cooldown,
    and auto-resolution.
    """

    def __init__(self, registry_path=REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.incidents = self._load_registry()

    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(
                    "Failed to load incident registry (%s) -- starting empty.", e
                )
        return {}

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.incidents, f, indent=2)

    def _get_key(self, unit_id: str, incident_type: str) -> str:
        return f"{unit_id}_{incident_type}"

    def get_active_incident(self, unit_id: str, incident_type: str) -> Optional[dict]:
        key      = self._get_key(unit_id, incident_type)
        incident = self.incidents.get(key)
        if incident and incident["status"] in [STATUS_OPEN, STATUS_IN_PROGRESS]:
            return incident
        return None

    def check_cooldown(self, unit_id: str, incident_type: str, new_severity: str) -> bool:
        key      = self._get_key(unit_id, incident_type)
        incident = self.incidents.get(key)
        if not incident:
            return False
        # Never suppress HIGH or CRITICAL severity escalations
        if SEVERITY_MAP.get(new_severity, "LOW") in ["HIGH", "CRITICAL"]:
            return False
        if incident["status"] in [STATUS_RESOLVED, STATUS_CLOSED]:
            updated_at = datetime.fromisoformat(incident["updated_at"])
            elapsed    = (datetime.now() - updated_at).total_seconds() / 3600
            if elapsed < COOLDOWN_HOURS:
                return True
        return False

    def update_or_create(
        self,
        unit_id: str,
        incident_type: str,
        ticket_data: dict,
    ) -> dict:
        key          = self._get_key(unit_id, incident_type)
        existing     = self.get_active_incident(unit_id, incident_type)
        now_str      = datetime.now().isoformat()
        new_severity = ticket_data["severity"]
        new_score    = ticket_data.get("layer2_result", {}).get("anomaly_pct", 0)

        if existing:
            old_severity = existing["severity"]
            escalated    = (
                compare_severity(new_severity, old_severity) == new_severity
                and new_severity != old_severity
            )
            if escalated:
                print(f"[TICKET ESCALATED] {existing['ticket_id']}: "
                      f"{old_severity} -> {new_severity}")
            else:
                print(f"[TICKET UPDATED] {existing['ticket_id']}")

            existing.update({
                "updated_at":      now_str,
                "severity":        compare_severity(old_severity, new_severity),
                "urgency":         ticket_data["urgency"],
                "last_anomaly_score": new_score,
                "latest_readings": ticket_data["sensor_snapshot"],
                "cost_escalation": ticket_data["cost_escalation"],
            })
            self.incidents[key] = existing
            self._save_registry()
            ticket_path = TICKETS_DIR / f"{existing['ticket_id']}.json"
            with open(ticket_path, "w") as f:
                json.dump(existing, f, indent=2)
            return existing
        else:
            print(f"[TICKET CREATED] {ticket_data['ticket_id']}: "
                  f"{incident_type} for {unit_id}")
            ticket_data["created_at"]         = now_str
            ticket_data["updated_at"]         = now_str
            ticket_data["last_anomaly_score"] = new_score
            self.incidents[key]               = ticket_data
            self._save_registry()
            return ticket_data

    def auto_resolve(self, unit_id: str, active_incident_types: list):
        for key, incident in self.incidents.items():
            if incident["unit_id"] != unit_id:
                continue
            if incident["status"] not in [STATUS_OPEN, STATUS_IN_PROGRESS]:
                continue
            if incident["incident_type"] not in active_incident_types:
                updated_at   = datetime.fromisoformat(incident["updated_at"])
                elapsed_mins = (datetime.now() - updated_at).total_seconds() / 60
                if elapsed_mins >= RECOVERY_MINUTES:
                    incident["status"]     = STATUS_RESOLVED
                    incident["updated_at"] = datetime.now().isoformat()
                    print(f"[TICKET RESOLVED] {incident['ticket_id']} — normal readings")
                    self._save_registry()
                    ticket_path = TICKETS_DIR / f"{incident['ticket_id']}.json"
                    if ticket_path.exists():
                        with open(ticket_path, "w") as f:
                            json.dump(incident, f, indent=2)


# ─────────────────────────────────────────────────────
# TICKET GENERATOR
# ─────────────────────────────────────────────────────

def generate_ticket(
    batch_df: pd.DataFrame,
    detection_result: dict,
    unit_id: str = "CARRIER-30XA-01",
) -> Optional[dict]:
    """
    Main entry point: generate or update a maintenance ticket.

    Parameters
    ----------
    batch_df         : raw sensor batch (60 rows)
    detection_result : output of detect_anomaly()
    unit_id          : fleet unit identifier

    Returns
    -------
    dict ticket, or None if filtered/suppressed.
    """
    severity       = detection_result.get("severity", "NORMAL")
    confidence_pct = detection_result.get("layer2", {}).get("anomaly_pct", 0)

    if severity == "NORMAL":
        return None

    # Confidence filters
    if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get("WARNING", 0):
        if confidence_pct < 20.0:
            return None
    if confidence_pct < MIN_CONFIDENCE_PCT:
        return None

    mgr = IncidentManager()

    # ── FIX: extract cause scores from result["scores"], not the top-level dict ──
    # calculate_freeze_risk() returns:
    #   { "total_freeze_risk": float,
    #     "dominant_cause":    str,
    #     "scores":            {"low_refrigerant": float, ...} }
    # Iterating FREEZE_CAUSE_WEIGHTS on the top-level dict caused KeyError.
    risk_rows = [calculate_freeze_risk(row) for row in batch_df.to_dict("records")]

    # Average each cause score across the batch
    avg_scores: dict = {
        cause: round(float(np.mean([r["scores"][cause] for r in risk_rows])), 2)
        for cause in FREEZE_CAUSE_WEIGHTS
    }

    # Average total risk across the batch
    avg_total: float = round(
        float(np.mean([r["total_freeze_risk"] for r in risk_rows])), 2
    )

    dominant      = get_dominant_cause(avg_scores)
    incident_type = f"THERMAL_{dominant.upper()}"

    if mgr.check_cooldown(unit_id, incident_type, severity):
        return None

    now       = datetime.now()
    ticket_id = (
        f"TKT-{unit_id}-{now.strftime('%Y%m%d%H%M')}-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )

    urgency     = estimate_urgency(severity, avg_total)
    action_plan = get_action_plan(dominant, severity)
    costs       = calculate_costs(severity, avg_total)

    raw_sensor_keys = list(CARRIER_BASELINE.keys())
    latest_row      = batch_df.iloc[-1].to_dict()
    sensor_snapshot = {
        k: round(float(latest_row.get(k, 0)), 4)
        for k in raw_sensor_keys if k in latest_row
    }
    z_snapshot = {
        k: round(float(latest_row.get(f"z_{k}", 0)), 3)
        for k in raw_sensor_keys
    }

    stage_labels = {
        "ADVISORY":         "EARLY STAGE",
        "WARNING":          "DEVELOPING STAGE",
        "CRITICAL":         "SERIOUS STAGE",
        "IMMINENT_FAILURE": "EMERGENCY STAGE",
    }
    stage = stage_labels.get(severity, "DETECTED")

    ticket = {
        "ticket_id":       ticket_id,
        "unit_id":         unit_id,
        "incident_type":   incident_type,
        "generated_at":    now.isoformat(),
        "status":          STATUS_OPEN,
        "severity":        severity,
        "mapped_severity": SEVERITY_MAP.get(severity, "LOW"),
        "urgency":         urgency,
        "freeze_risk_pct": avg_total,
        "dominant_cause":  dominant,
        "cause_scores":    avg_scores,      # full breakdown for UI / audit
        "action_plan":     action_plan,
        "cost_escalation": costs,
        "sensor_snapshot": sensor_snapshot,
        "z_score_snapshot": z_snapshot,
        "layer1_result": {
            "flagged_sensors": detection_result["layer1"]["flagged_sensors"],
            "max_z_score":     detection_result["layer1"]["max_z_score"],
            "worst_severity":  detection_result["layer1"]["worst_severity"],
        },
        "layer2_result": {
            "anomaly_pct": detection_result["layer2"]["anomaly_pct"],
            "confirmed":   detection_result["layer2"]["confirmed"],
        },
    }

    final_ticket = mgr.update_or_create(unit_id, incident_type, ticket)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n[TICKET] {final_ticket['ticket_id']}")
    print(f"         Severity  : {final_ticket['severity']}")
    print(f"         Detected  : {stage} — {final_ticket['urgency']['display']} before failure")
    print(f"         Cause     : {final_ticket['dominant_cause']} "
          f"({round(avg_scores[dominant], 1)}%)")
    print(f"         Urgency   : {final_ticket['urgency']['display']} "
          f"({final_ticket['urgency']['urgency_band']})")
    print(f"         Cost now  : Rs {final_ticket['cost_escalation']['repair_cost_now_inr']:,}")
    print(f"         Cost 4wk  : Rs {final_ticket['cost_escalation']['repair_cost_if_ignored_4weeks_inr']:,}")
    print(f"         Daily loss: Rs {final_ticket['cost_escalation']['daily_energy_excess_inr']:,}/day")

    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TICKETS_DIR / f"{final_ticket['ticket_id']}.json", "w") as f:
        json.dump(final_ticket, f, indent=2)

    return final_ticket