"""
batch_pipeline.py

Main orchestrator for Carrier HVAC Predictive Maintenance.
Now upgraded with Fleet Management and Sensor Health Checks.

Commands:
  generate   Generate synthetic fleet data (10 units, 480 min)
  train      Train Isolation Forest on normal-scenario data
  fleet      Process all units through sensor health + anomaly detection
  report     Generate fleet-wide savings report
  nightly    Sync parquets to SQLite, run TTL cleanup
  demo       End-to-end: generate → train → fleet (for presentations)

FIX 1 — run_ttl_cleanup argument mismatch:
  run_nightly() previously called run_ttl_cleanup(RAW_DIR), but
  ttl_manager.run_ttl_cleanup() takes only an optional dry_run bool —
  it reads its own TTL_LOG_PATH internally.  Passing RAW_DIR was either
  silently ignored (if the function accepted *args) or raised TypeError.
  Fixed to call run_ttl_cleanup() with no positional argument.

FIX 2 — savings report undercounts early detections:
  generate_savings_report() in fleet_manager.py only counted ADVISORY
  and WARNING tickets as "caught early", excluding CRITICAL tickets that
  prevented IMMINENT_FAILURE escalation.  The batch pipeline's own
  reporting helper now counts any ticket whose severity is below
  IMMINENT_FAILURE as an early detection, which is the correct business
  interpretation (anything caught before compressor damage = early).

FIX 3 — NORMAL units in fleet display:
  Units with severity NORMAL have a priority_score of 0 (correct by
  design) but were being mixed into the ranked table with no visual
  distinction.  The fleet batch runner now separates the queue into
  attention-needed and healthy sections when printing, so the dashboard
  is not cluttered by 0-score rows.
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from carrier_baseline    import TTL_DAYS, SEVERITY_ORDER
from hvac_simulator      import (generate_dataset, ALL_SCENARIOS,
                                 generate_fleet_data, FLEET_CONFIG)
from feature_engineering import engineer_features, get_model_features
from ttl_manager         import (register_file_ttl, extend_ttl_on_anomaly,
                                 run_ttl_cleanup)           # ← FIX 1: no args in call site
from db_manager          import (init_db, merge_parquets_to_db,
                                 get_normal_training_data,
                                 run_db_ttl_cleanup, get_db_stats)
from anomaly_detector    import (RuleEngine, IsolationForestDetector,
                                 detect_anomaly)
from ticket_engine       import generate_ticket
from sensor_health       import SensorHealthChecker
from fleet_manager       import FleetManager

# ─────────────────────────────────────────────────────
# PATH CONSTANTS
# ─────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
DB_PATH     = "data/hvac_sensor.db"
MODEL_DIR   = "models"
STATE_FILE  = "data/pipeline_state.json"
TICKETS_DIR = Path("alerts/tickets")


# ─────────────────────────────────────────────────────
# PIPELINE STATE
# ─────────────────────────────────────────────────────

class PipelineState:
    """
    Persistent pipeline state: batch tails, processed filenames, counters.

    Serialised as JSON (not pickle) for safety and forward-compatibility.
    schema_version guards against loading state written by an older
    version of this class with different field names.
    """
    SCHEMA_VERSION = 1

    def __init__(self):
        self.previous_batch_tails: dict = {}   # unit_id → list[dict] (last 30 rows)
        self.processed_files: set       = set()
        self.batch_counters: dict       = {}

    def get_tail(self, unit_id: str) -> Optional[pd.DataFrame]:
        records = self.previous_batch_tails.get(unit_id)
        if records is None:
            return None
        df = pd.DataFrame(records)
        # Re-parse timestamps that were serialised to strings by JSON
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def update_tail(self, unit_id: str, batch_df: pd.DataFrame):
        self.previous_batch_tails[unit_id] = (
            batch_df.tail(30).copy().to_dict("records")
        )

    def mark_processed(self, filepath: str):
        self.processed_files.add(Path(filepath).name)

    def is_processed(self, filepath: str) -> bool:
        return Path(filepath).name in self.processed_files

    def save(self, path: str = STATE_FILE):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version":       self.SCHEMA_VERSION,
            "previous_batch_tails": self.previous_batch_tails,
            "processed_files":      list(self.processed_files),
            "batch_counters":       self.batch_counters,
        }
        tmp = Path(path).with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)   # atomic on POSIX

    @classmethod
    def load(cls, path: str = STATE_FILE) -> "PipelineState":
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if data.get("schema_version") != cls.SCHEMA_VERSION:
                    logging.warning(
                        "PipelineState schema mismatch (got %s, want %s) "
                        "-- resetting to empty state.",
                        data.get("schema_version"), cls.SCHEMA_VERSION,
                    )
                    return cls()
                obj = cls()
                obj.previous_batch_tails = data.get("previous_batch_tails", {})
                obj.processed_files      = set(data.get("processed_files", []))
                obj.batch_counters       = data.get("batch_counters", {})
                return obj
            except Exception as e:
                logging.warning(
                    "Failed to load pipeline state (%s) -- starting fresh.", e
                )
        return cls()


# ─────────────────────────────────────────────────────
# SINGLE-BATCH PROCESSOR
# ─────────────────────────────────────────────────────

def run_batch(
    parquet_path: str,
    unit_id: str,
    model_dir: str,
    state: PipelineState,
) -> dict:
    """Process one parquet batch through sensor health and anomaly detection."""
    raw_df = pd.read_parquet(parquet_path)
    if raw_df.empty:
        return {"status": "empty"}

    # ── STEP 1: Sensor health check ───────────────────────────────────────────
    sensor_checker = SensorHealthChecker()
    health_result  = sensor_checker.check_batch(raw_df)

    if health_result["recommendation"] == "SKIP_ANOMALY_DETECTION":
        ticket = sensor_checker.generate_sensor_fault_ticket(
            unit_id, health_result["fault_details"]
        )
        print(f"[SENSOR FAULT] {unit_id}: {health_result['faulty_sensors']}")
        print("  -> Anomaly detection SKIPPED")
        if ticket:
            print(f"  -> Sensor ticket: {ticket['ticket_id']}")
        return {
            "status":           "SENSOR_FAULT",
            "ticket":           ticket,
            "anomaly_detected": False,
            "severity":         ticket["severity"] if ticket else "CRITICAL",
        }

    if health_result["recommendation"] == "PROCEED_WITH_CAUTION":
        print(f"[SENSOR WARNING] {unit_id}: degraded sensor — proceeding with caution")

    # ── STEP 2: Log defrost rows ──────────────────────────────────────────────
    defrost_count = (
        int(raw_df["is_defrost_cycle"].sum())
        if "is_defrost_cycle" in raw_df.columns
        else 0
    )
    if defrost_count > 0:
        print(
            f"  [DEFROST] {unit_id}: {defrost_count} defrost rows "
            f"will be excluded from anomaly check"
        )

    # ── STEP 3: Feature engineering + anomaly detection ───────────────────────
    previous_tail = state.get_tail(unit_id)
    feat_df       = engineer_features(raw_df, previous_tail=previous_tail)
    state.update_tail(unit_id, feat_df)

    result = detect_anomaly(raw_df, previous_tail=previous_tail, model_dir=model_dir)

    ticket = None
    if result["confirmed"]:
        print(f"[ANOMALY] {result['severity']}  {unit_id} | {Path(parquet_path).name}")
        extend_ttl_on_anomaly(parquet_path, result["severity"])
        ticket = generate_ticket(raw_df, result, unit_id=unit_id)
        if ticket:
            print(f"  -> Ticket: {ticket['ticket_id']}")
            print(
                f"     Urgency: {ticket['urgency']['hours']} hrs | "
                f"Cost/day: Rs {ticket['cost_escalation']['cost_per_day_delay_inr']:,}"
            )
    else:
        print(f"[{unit_id}] {result['decision']} — TTL 2 days")

    state.mark_processed(parquet_path)
    return {
        "status":           result["decision"],
        "severity":         result["severity"],
        "ticket":           ticket,
        "anomaly_detected": result["confirmed"],
    }


# ─────────────────────────────────────────────────────
# FLEET BATCH RUNNER
# ─────────────────────────────────────────────────────

def run_fleet_batch(data_dir: str = RAW_DIR, model_dir: str = MODEL_DIR):
    """Process all units in the fleet, then print a prioritised dashboard."""
    print("\n[FLEET BATCH] Processing all units...")
    fleet = FleetManager()
    state = PipelineState.load()

    unit_dirs = sorted(
        d for d in Path(data_dir).iterdir()
        if d.is_dir() and d.name.startswith("CARRIER")
    )
    if not unit_dirs:
        print("[FLEET] No unit data found. Run 'generate' first.")
        return

    for unit_dir in unit_dirs:
        unit_id  = unit_dir.name
        print(f"\n[{unit_id}] " + "-" * 40)

        parquets         = sorted(unit_dir.glob("**/*.parquet"))
        active_incidents = []

        for p in parquets:
            if state.is_processed(str(p)):
                continue
            res    = run_batch(str(p), unit_id, model_dir, state)
            ticket = res.get("ticket")
            if ticket:
                active_incidents.append(ticket["incident_type"])
                fleet.update_unit(unit_id, ticket)

        # Auto-resolve incidents that have been quiet long enough
        from ticket_engine import IncidentManager
        IncidentManager().auto_resolve(unit_id, active_incidents)

    state.save()

    # ── FIX 3: Print attention-needed units separately from healthy ones ──────
    _print_fleet_dashboard_enhanced(fleet)
    return fleet.get_fleet_summary()


def _print_fleet_dashboard_enhanced(fleet: FleetManager) -> None:
    """
    Print fleet dashboard with healthy units separated from those needing
    attention.  Units with priority_score == 0 (NORMAL severity) are
    listed in a separate 'Healthy' section so they don't clutter the
    priority queue or mislead on urgency ranking.
    """
    queue   = fleet.get_dispatch_queue()
    summary = fleet.get_fleet_summary()

    attention = [q for q in queue if q["priority_score"] > 0]
    healthy   = [q for q in queue if q["priority_score"] == 0]

    print("\n+--------------------------------------------------------------------+")
    print(f"|          CARRIER FLEET STATUS — {summary['total_units']} Units                      |")
    print("+------+------------------+----------+------------+------------------+")
    print("| Rank | Unit ID          | Severity | Lead Time  | Action           |")
    print("+------+------------------+----------+------------+------------------+")

    if attention:
        for item in attention[:10]:
            print(
                f"|  {item['rank']:<3} | {item['unit_id']:<16} | "
                f"{item['severity']:<8} | {item['urgency_display']:>10} | "
                f"{item['action']:<16} |"
            )
    else:
        print("|  --  | (no units require attention)                               |")

    if healthy:
        print("+--------------------------------------------------------------------+")
        print(f"|  Healthy units ({len(healthy)}): " +
              ", ".join(h["unit_id"] for h in healthy) + " " * 10 + "|")

    print("+--------------------------------------------------------------------+")
    print(f"Fleet Health Score : {summary['fleet_health_score']}/100")
    print(f"Total Daily Loss   : Rs {summary['total_daily_loss_inr']:,}")
    print(f"Units Needing Attention: {len(attention)}")


# ─────────────────────────────────────────────────────
# SAVINGS REPORT  (FIX 2 — counts CRITICAL as early too)
# ─────────────────────────────────────────────────────

def generate_pipeline_savings_report() -> dict:
    """
    Estimate maintenance savings from early detection across all tickets.

    FIX 2: The original fleet_manager version only counted ADVISORY and
    WARNING tickets as "early detections", missing CRITICAL tickets that
    prevented an IMMINENT_FAILURE / compressor damage event.  The correct
    business logic is: any ticket whose severity is strictly below
    IMMINENT_FAILURE counts as early detection (i.e. we caught it before
    the catastrophic outcome).

    Emergency cost baseline: Rs 74,000 (compressor call-out + parts).
    Compressor replacement if freeze damage: Rs 1,20,000+.
    We conservatively use the call-out figure.
    """
    EMERGENCY_COST = 74_000

    if not TICKETS_DIR.exists():
        return {"error": "No tickets directory found"}

    # De-duplicate: keep the most recent ticket per unit
    latest_tickets: dict = {}
    for tf in TICKETS_DIR.glob("*.json"):
        try:
            with open(tf) as f:
                t = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        unit_id  = t.get("unit_id", "unknown")
        gen_at   = t.get("generated_at", "")
        if unit_id not in latest_tickets or gen_at > latest_tickets[unit_id].get("generated_at", ""):
            latest_tickets[unit_id] = t

    events_caught_early = 0
    total_actual_cost   = 0

    for t in latest_tickets.values():
        severity = t.get("severity", "NORMAL")
        # FIX 2: CRITICAL is also an early detection (before compressor damage)
        if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get("IMMINENT_FAILURE", 4):
            if severity != "NORMAL":
                events_caught_early += 1

        cost_now = t.get("cost_escalation", {}).get("repair_cost_now_inr", 25_000)
        total_actual_cost += cost_now

    emergency_cost_avoided = events_caught_early * EMERGENCY_COST
    net_savings            = emergency_cost_avoided - total_actual_cost

    report = {
        "events_detected_early":    events_caught_early,
        "emergency_cost_avoided":   emergency_cost_avoided,
        "actual_maintenance_cost":  total_actual_cost,
        "net_savings":              net_savings,
    }

    print(f"\nEvents detected early   : {events_caught_early}")
    print(f"Emergency cost avoided  : Rs {emergency_cost_avoided:,}")
    print(f"Actual maintenance cost : Rs {total_actual_cost:,}")
    print(f"Net savings             : Rs {net_savings:,}")

    return report


# ─────────────────────────────────────────────────────
# NIGHTLY MAINTENANCE  (FIX 1 — correct TTL call)
# ─────────────────────────────────────────────────────

def run_nightly():
    """
    Nightly maintenance window: DB sync + TTL cleanup.

    FIX 1: run_ttl_cleanup() accepts only an optional dry_run bool.
    The previous call run_ttl_cleanup(RAW_DIR) passed a string as the
    first positional argument, which either caused a TypeError or was
    silently cast to truthy (dry_run=True), meaning files were never
    actually deleted.  Fixed to call with no positional argument.
    """
    print("\n[NIGHTLY] Starting maintenance...")
    init_db(DB_PATH)
    merge_parquets_to_db(RAW_DIR, DB_PATH)
    run_db_ttl_cleanup(DB_PATH)
    run_ttl_cleanup()                       # ← FIX 1: no RAW_DIR argument
    print("[NIGHTLY] Maintenance complete.")


# ─────────────────────────────────────────────────────
# DEMO (end-to-end for presentations)
# ─────────────────────────────────────────────────────

def cmd_demo():
    """End-to-end demo: generate → train → fleet batch → savings report."""
    print("\n[FLEET] Generating data for 10 units...")
    generate_fleet_data(total_minutes=480)

    print("\n[TRAIN] Training IsolationForest on normal data...")
    normal_parquets = list(Path(RAW_DIR).glob("**/normal/*.parquet"))
    if not normal_parquets:
        print("[TRAIN] ERROR: No normal data found for training.")
        return
    normal_df = pd.concat([pd.read_parquet(f) for f in normal_parquets])
    detector  = IsolationForestDetector(model_dir=MODEL_DIR)
    detector.train(normal_df)

    run_fleet_batch()
    generate_pipeline_savings_report()


# ─────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python batch_pipeline.py "
            "[generate|train|run|fleet|report|nightly|demo]"
        )
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "generate":
        generate_fleet_data(total_minutes=480)

    elif cmd == "train":
        normal_parquets = list(Path(RAW_DIR).glob("**/normal/*.parquet"))
        if not normal_parquets:
            print("[TRAIN] No normal data found.")
            sys.exit(1)
        normal_df = pd.concat([pd.read_parquet(f) for f in normal_parquets])
        IsolationForestDetector(model_dir=MODEL_DIR).train(normal_df)

    elif cmd == "fleet":
        run_fleet_batch()

    elif cmd == "report":
        generate_pipeline_savings_report()

    elif cmd == "nightly":
        run_nightly()

    elif cmd == "demo":
        cmd_demo()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()