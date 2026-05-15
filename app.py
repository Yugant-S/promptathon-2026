"""
app.py — HVAC Predictive Maintenance Dashboard API
Serves all data from existing project files.
"""

import json
import os
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, make_response, request

app = Flask(__name__, static_folder="static", static_url_path="")

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TICKETS_DIR = BASE_DIR / "alerts" / "tickets"
REGISTRY_PATH = DATA_DIR / "incident_registry.json"

# ── FREEZE CAUSE WEIGHTS (from carrier_baseline.py) ─────────────────────────
FREEZE_CAUSE_WEIGHTS = {
    "low_refrigerant":   0.40,
    "dirty_coil":        0.25,
    "exv_malfunction":   0.15,
    "low_chilled_water": 0.12,
    "condenser_issue":   0.08,
}

CAUSE_META = {
    "low_refrigerant":   {"label": "Low Refrigerant Charge",    "desc": "Suction pressure below threshold — charge depletion detected", "icon": "❄"},
    "dirty_coil":        {"label": "Dirty Evaporator Coil",     "desc": "Coil ΔT dropped — fouling / ice bridging detected",           "icon": "🌫"},
    "exv_malfunction":   {"label": "EXV Malfunction",           "desc": "Superheat oscillating ±4°C — valve hunting pattern",          "icon": "⚡"},
    "low_chilled_water": {"label": "Low Chilled Water Flow",    "desc": "Flow at 38 LPS — pump strainer suspect",                      "icon": "💧"},
    "condenser_issue":   {"label": "Condenser Heat Rejection",  "desc": "Discharge pressure elevated — borderline condenser fouling",  "icon": "🔥"},
}

ENGINEERS = [
    {"id": "ENG-001", "name": "Rajesh Sharma",   "specialization": "Refrigeration Systems"},
    {"id": "ENG-002", "name": "Priya Mehta",     "specialization": "Electrical & Controls"},
    {"id": "ENG-003", "name": "Anil Patel",      "specialization": "Mechanical / HVAC"},
    {"id": "ENG-004", "name": "Sunita Rao",      "specialization": "Condenser & Heat Transfer"},
    {"id": "ENG-005", "name": "Vikram Nair",     "specialization": "Chilled Water Systems"},
]

CARRIER_BASELINE = {
    "suction_pressure_bar":   {"mean": 4.2,  "std": 0.15, "unit": "bar",  "label": "Suction Pressure",    "critical_low": 3.50},
    "discharge_pressure_bar": {"mean": 11.0, "std": 0.30, "unit": "bar",  "label": "Discharge Pressure",  "critical_high": 13.50},
    "coil_delta_T_C":         {"mean": 8.0,  "std": 0.40, "unit": "°C",   "label": "Coil ΔT",             "warning_low": 7.0},
    "superheat_C":            {"mean": 6.0,  "std": 0.80, "unit": "°C",   "label": "Superheat",           "warning_low": 4.0},
    "subcooling_C":           {"mean": 5.0,  "std": 0.50, "unit": "°C",   "label": "Subcooling",          "warning_low": 3.75},
    "chilled_water_flow_LPS": {"mean": 55.0, "std": 2.0,  "unit": "LPS",  "label": "Chilled Water Flow",  "warning_low": 50.0},
    "compressor_current_A":   {"mean": 38.0, "std": 1.50, "unit": "A",    "label": "Compressor Current",  "critical_high": 42.5},
    "blower_current_A":       {"mean": 12.5, "std": 0.30, "unit": "A",    "label": "Blower Current",      "warning_high": 13.25},
}

GLOBAL_REGISTRY = None

def load_registry():
    global GLOBAL_REGISTRY
    if GLOBAL_REGISTRY is None:
        try:
            with open(REGISTRY_PATH) as f:
                GLOBAL_REGISTRY = json.load(f)
        except Exception as e:
            GLOBAL_REGISTRY = {}
    return GLOBAL_REGISTRY

def sensor_status(key, value):
    b = CARRIER_BASELINE.get(key, {})
    mean = b.get("mean", value)
    std  = b.get("std", 1)
    z = abs(value - mean) / max(std, 0.01)
    if z > 3.0:   return "CRITICAL"
    if z > 2.0:   return "WARNING"
    return "NORMAL"

def build_unit(unit_id, incident):
    readings = incident.get("latest_readings") or incident.get("sensor_snapshot") or {}
    cause = incident.get("dominant_cause", "low_refrigerant")
    freeze_risk = incident.get("freeze_risk_pct", 0)
    urgency_hrs = incident.get("urgency", {}).get("hours", 48)
    anomaly_pct = incident.get("last_anomaly_score", 0)

    # Cause contributions — weight-based, normalised
    total_w = sum(FREEZE_CAUSE_WEIGHTS.values())
    causes = []
    for c, w in FREEZE_CAUSE_WEIGHTS.items():
        boost = 1.8 if c == cause else 1.0
        pct = round((w * boost / (total_w + w * 0.8)) * 100, 1)
        causes.append({**CAUSE_META[c], "key": c, "pct": pct})
    # Normalise to 100
    total_pct = sum(x["pct"] for x in causes)
    for c in causes:
        c["pct"] = round(c["pct"] / total_pct * 100, 1)
    causes.sort(key=lambda x: -x["pct"])

    # Sensor cards
    sensors = []
    for key, meta in CARRIER_BASELINE.items():
        val = readings.get(key)
        if val is not None:
            sensors.append({
                "key":    key,
                "label":  meta["label"],
                "value":  round(val, 2),
                "unit":   meta["unit"],
                "status": sensor_status(key, val),
            })

    # Cost
    cost = incident.get("cost_escalation", {})
    cost_per_day = cost.get("cost_per_day_delay_inr", 0)

    # Status text
    sev = incident.get("severity", "ADVISORY")
    status_label = {
        "IMMINENT_FAILURE": "Degrading",
        "CRITICAL":         "Critical",
        "WARNING":          "Warning",
        "ADVISORY":         "Advisory",
    }.get(sev, "Unknown")

    return {
        "unit_id":        unit_id,
        "incident_type":  incident.get("incident_type", ""),
        "status":         status_label,
        "severity":       sev,
        "mapped_severity": incident.get("mapped_severity", "CRITICAL"),
        "freeze_risk_pct": freeze_risk,
        "anomaly_pct":    anomaly_pct,
        "urgency_hours":  urgency_hrs,
        "urgency_display": incident.get("urgency", {}).get("display", "2 days"),
        "dominant_cause": cause,
        "cost_per_day":   cost_per_day,
        "action_plan":    incident.get("action_plan", []),
        "causes":         causes,
        "sensors":        sensors,
        "cost_escalation": cost,
    }

def build_ticket(unit_id, incident, idx):
    eng = incident.get("assigned_engineer_override", ENGINEERS[idx % len(ENGINEERS)])
    sev = incident.get("mapped_severity", "CRITICAL")
    sla_hrs = {"CRITICAL": 4, "HIGH": 24, "MEDIUM": 72, "LOW": 168}.get(sev, 48)
    created = datetime.fromisoformat(incident.get("created_at", datetime.now().isoformat()))
    elapsed = (datetime.now() - created).total_seconds() / 3600
    sla_remaining = max(0, sla_hrs - elapsed)

    return {
        "ticket_id":       incident.get("ticket_id", f"TKT-{unit_id}"),
        "unit_id":         unit_id,
        "incident_type":   incident.get("incident_type", ""),
        "status":          incident.get("status", "OPEN"),
        "severity":        sev,
        "urgency_band":    incident.get("urgency", {}).get("urgency_band", "CRITICAL"),
        "urgency_message": incident.get("urgency", {}).get("message", ""),
        "freeze_risk_pct": incident.get("freeze_risk_pct", 0),
        "dominant_cause":  incident.get("dominant_cause", "unknown"),
        "cause_label":     CAUSE_META.get(incident.get("dominant_cause",""), {}).get("label", "Unknown"),
        "action_plan":     incident.get("action_plan", []),
        "assigned_engineer": eng,
        "sla_hours":       sla_hrs,
        "sla_remaining_hours": round(sla_remaining, 1),
        "anomaly_pct":     incident.get("last_anomaly_score", 0),
        "max_z_score":     incident.get("layer1_result", {}).get("max_z_score", 0),
        "escalation_level": "L3" if sev == "CRITICAL" else "L2",
        "cost_per_day":    incident.get("cost_escalation", {}).get("cost_per_day_delay_inr", 0),
        "generated_at":    incident.get("generated_at", ""),
        "updated_at":      incident.get("updated_at", ""),
    }

# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/fleet")
def fleet():
    registry = load_registry()
    units = []
    for i, (key, incident) in enumerate(registry.items()):
        unit_id = incident.get("unit_id", key.split("_")[0])
        units.append(build_unit(unit_id, incident))
    # Add normal units (03, 07, 09)
    for uid in ["CARRIER-30XA-03", "CARRIER-30XA-07", "CARRIER-30XA-09"]:
        units.append({
            "unit_id": uid,
            "incident_type": "NORMAL",
            "status": "Nominal",
            "severity": "NORMAL",
            "mapped_severity": "LOW",
            "freeze_risk_pct": round(random.uniform(2, 8), 2),
            "anomaly_pct": round(random.uniform(5, 20), 1),
            "urgency_hours": 720,
            "urgency_display": "30 days",
            "dominant_cause": None,
            "cost_per_day": 0,
            "action_plan": ["No action required — unit operating within normal parameters."],
            "causes": [],
            "sensors": [
                {"key": "suction_pressure_bar", "label": "Suction Pressure", "value": round(random.uniform(4.0, 4.4), 2), "unit": "bar", "status": "NORMAL"},
                {"key": "discharge_pressure_bar", "label": "Discharge Pressure", "value": round(random.uniform(10.6, 11.4), 2), "unit": "bar", "status": "NORMAL"},
                {"key": "compressor_current_A", "label": "Compressor Current", "value": round(random.uniform(36, 40), 1), "unit": "A", "status": "NORMAL"},
            ],
            "cost_escalation": {}
        })
    summary = {
        "total_units": len(units),
        "critical_count": sum(1 for u in units if u["severity"] in ("IMMINENT_FAILURE", "CRITICAL")),
        "warning_count": sum(1 for u in units if u["severity"] == "WARNING"),
        "normal_count": sum(1 for u in units if u["severity"] in ("NORMAL", "ADVISORY")),
        "total_daily_loss_inr": sum(u["cost_per_day"] for u in units),
    }
    return jsonify({"units": units, "summary": summary})

@app.route("/api/tickets")
def tickets():
    registry = load_registry()
    result = []
    for i, (key, incident) in enumerate(registry.items()):
        unit_id = incident.get("unit_id", key.split("_")[0])
        result.append(build_ticket(unit_id, incident, i))
    result.sort(key=lambda t: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(t["severity"], 4))
    return jsonify({"tickets": result, "count": len(result)})

@app.route("/api/unit/<unit_id>")
def unit_detail(unit_id):
    registry = load_registry()
    for key, incident in registry.items():
        if incident.get("unit_id") == unit_id:
            return jsonify(build_unit(unit_id, incident))
    return jsonify({"error": "Unit not found"}), 404

@app.route("/api/engineers")
def get_engineers():
    return jsonify({"engineers": ENGINEERS})

@app.route("/api/tickets/<ticket_id>", methods=["PUT"])
def update_ticket(ticket_id):
    registry = load_registry()
    data = request.json
    for key, incident in registry.items():
        tid = incident.get("ticket_id", f"TKT-{incident.get('unit_id', key.split('_')[0])}")
        if tid == ticket_id:
            if "status" in data:
                incident["status"] = data["status"]
            if "assigned_engineer" in data:
                eng_id = data["assigned_engineer"]
                for eng in ENGINEERS:
                    if eng["id"] == eng_id:
                        incident["assigned_engineer_override"] = eng
                        break
            # In a real app we'd also store an activity log here
            return jsonify({"status": "success"})
    return jsonify({"error": "Not found"}), 404

@app.route("/api/alerts")
def alerts():
    registry = load_registry()
    alerts_list = []
    for key, incident in registry.items():
        uid = incident.get("unit_id", "")
        sev = incident.get("mapped_severity", "CRITICAL")
        cause = incident.get("dominant_cause", "")
        alerts_list.append({
            "id":        f"ALT-{uid}",
            "unit_id":   uid,
            "severity":  sev,
            "message":   f"{CAUSE_META.get(cause, {}).get('label', cause)} detected on {uid}",
            "detail":    incident.get("urgency", {}).get("message", ""),
            "timestamp": incident.get("generated_at", ""),
            "acknowledged": False,
        })
    return jsonify({"alerts": alerts_list})

@app.route("/api/trends/<unit_id>")
def trends(unit_id):
    """Generate synthetic trend data for charts (based on sensor snapshot)."""
    registry = load_registry()
    incident = None
    for key, v in registry.items():
        if v.get("unit_id") == unit_id:
            incident = v
            break
    if not incident:
        return jsonify({"error": "Not found"}), 404

    snap = incident.get("sensor_snapshot", {})
    latest = incident.get("latest_readings") or snap
    freeze_risk = incident.get("freeze_risk_pct", 10)

    # Generate 24-point trend (last 24 hours)
    points = []
    for i in range(24):
        t = (datetime.now() - timedelta(hours=23-i)).strftime("%H:%M")
        noise = random.uniform(-0.15, 0.15)
        drift = i / 24 * 0.3  # worsening trend
        fr = max(0, min(100, freeze_risk - (23-i) * 0.4 + random.uniform(-2, 2)))
        points.append({
            "time": t,
            "freeze_risk": round(fr, 1),
            "suction_pressure": round((latest.get("suction_pressure_bar", 4.2)) - drift + noise, 2),
            "discharge_pressure": round((latest.get("discharge_pressure_bar", 11.0)) + drift*2 + noise, 2),
            "superheat": round((latest.get("superheat_C", 6.0)) + drift + noise, 2),
            "anomaly_score": round(min(100, (incident.get("last_anomaly_score", 50)) * (i/24)*1.2), 1),
        })
    return jsonify({"unit_id": unit_id, "trends": points})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
