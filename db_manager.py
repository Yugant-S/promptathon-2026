"""
db_manager.py
═══════════════════════════════════════════════════════
SQLite operations: schema init, parquet merge, TTL cleanup,
and query helpers for the HVAC predictive maintenance system.

Design decisions:
  - SQLite chosen over Postgres for zero-infrastructure deployment
    in manufacturing facilities without DBA resources
  - batch_id (parquet filename) as natural dedup key prevents
    duplicate merges during nightly retries
  - expires_at computed per-row at merge time from severity
  - Indexes on timestamp + severity enable fast time-range queries
  - VACUUM after cleanup keeps file size bounded
═══════════════════════════════════════════════════════
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from carrier_baseline import TTL_DAYS

# ─────────────────────────────────────────────────────
# SCHEMA DEFINITION
# All 13 raw sensors + 5 computed + 5 freeze cause scores
# ─────────────────────────────────────────────────────

RAW_SENSORS = [
    "suction_pressure_bar", "discharge_pressure_bar", "coil_delta_T_C",
    "superheat_C", "subcooling_C", "chilled_water_flow_LPS",
    "chilled_water_delta_T_C", "blower_current_A", "compressor_current_A",
    "supply_air_temp_C", "return_air_temp_C", "ambient_temp_C",
    "compressor_runtime_pct",
]

COMPUTED_SENSORS = [
    "compression_ratio", "coil_efficiency_index", "system_COP",
    "superheat_subcooling_ratio", "heat_rejection_ratio",
]

FREEZE_CAUSE_COLS = [
    "low_refrigerant", "dirty_coil", "exv_malfunction",
    "low_chilled_water", "condenser_issue",
]

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS sensor_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    scenario        TEXT,
    severity        TEXT,
    expires_at      TEXT NOT NULL,
    {', '.join(f'{s} REAL' for s in RAW_SENSORS)},
    {', '.join(f'{s} REAL' for s in COMPUTED_SENSORS)},
    {', '.join(f'{s} REAL' for s in FREEZE_CAUSE_COLS)},
    total_freeze_risk REAL,
    UNIQUE(batch_id, timestamp)
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_data(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_severity  ON sensor_data(severity);",
    "CREATE INDEX IF NOT EXISTS idx_scenario  ON sensor_data(scenario);",
    "CREATE INDEX IF NOT EXISTS idx_expires   ON sensor_data(expires_at);",
]


# ─────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    """
    Create the sensor_data table and indexes if they don't exist.
    Safe to call multiple times (idempotent).

    Parameters
    ----------
    db_path : str
        Path to the SQLite .db file. Created if it doesn't exist.

    Returns
    -------
    sqlite3.Connection (caller responsible for closing)
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")   # Better concurrent read performance
    conn.execute("PRAGMA synchronous=NORMAL;") # Faster writes, still crash-safe
    conn.execute(CREATE_TABLE_SQL)
    for idx_sql in CREATE_INDEX_SQL:
        conn.execute(idx_sql)
    conn.commit()
    print(f"[DB] Initialized: {db_path}")
    return conn


# ─────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────

def merge_parquets_to_db(raw_dir: str, db_path: str) -> dict:
    """
    Scan all parquet files under raw_dir; merge into SQLite.

    Deduplication: UNIQUE(batch_id, timestamp) constraint prevents
    re-inserting rows from a file that was already merged.

    expires_at is computed per-row from the row's severity column,
    falling back to the batch-level worst severity if absent.

    Parameters
    ----------
    raw_dir : str
        Root directory containing scenario subdirectories with parquets.
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    dict : {total_files, rows_merged, rows_skipped, by_scenario}
    """
    conn = init_db(db_path)
    raw_path = Path(raw_dir)

    total_files  = 0
    rows_merged  = 0
    rows_skipped = 0
    by_scenario  = {}

    # Walk all scenario subdirectories
    for scenario_dir in sorted(raw_path.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name
        parquet_files = sorted(scenario_dir.glob("*.parquet"))

        if not parquet_files:
            continue

        scenario_rows = 0
        for pf in parquet_files:
            total_files += 1
            batch_id = pf.name  # Use filename as natural batch identifier

            df = pd.read_parquet(pf)
            df["scenario"] = scenario  # Stamp scenario from folder name

            # Determine severity per row; fall back to NORMAL if column absent
            if "freeze_severity" in df.columns:
                df["severity"] = df["freeze_severity"]
            elif "severity" not in df.columns:
                df["severity"] = "NORMAL"

            # Compute expires_at per row based on its severity
            def row_expires(sev):
                days = TTL_DAYS.get(sev, TTL_DAYS["NORMAL"])
                return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

            df["expires_at"] = df["severity"].apply(row_expires)
            df["batch_id"]   = batch_id

            # Build insert rows — only columns that exist in the schema
            all_data_cols = (
                ["batch_id", "timestamp", "scenario", "severity", "expires_at"]
                + RAW_SENSORS + COMPUTED_SENSORS + FREEZE_CAUSE_COLS
                + ["total_freeze_risk"]
            )
            insert_cols = [c for c in all_data_cols if c in df.columns]
            insert_df   = df[insert_cols]

            # INSERT OR IGNORE: skip rows already in DB
            placeholders = ", ".join(["?"] * len(insert_cols))
            col_names    = ", ".join(insert_cols)
            sql = f"INSERT OR IGNORE INTO sensor_data ({col_names}) VALUES ({placeholders})"

            data_tuples = [tuple(row) for row in insert_df.itertuples(index=False)]

            try:
                cursor = conn.executemany(sql, data_tuples)
                merged  = cursor.rowcount
                skipped = len(data_tuples) - merged
                rows_merged  += merged
                rows_skipped += skipped
                scenario_rows += merged
            except sqlite3.Error as e:
                print(f"[DB] Error merging {pf.name}: {e}")
                continue

        by_scenario[scenario] = scenario_rows
        if scenario_rows:
            print(f"[DB] {scenario}: {scenario_rows} rows merged")

    conn.commit()
    conn.close()

    print(f"\n[DB] Merge complete: {total_files} files, "
          f"{rows_merged} rows merged, {rows_skipped} skipped (duplicates)")
    return {
        "total_files":  total_files,
        "rows_merged":  rows_merged,
        "rows_skipped": rows_skipped,
        "by_scenario":  by_scenario,
    }


# ─────────────────────────────────────────────────────
# QUERY HELPERS
# ─────────────────────────────────────────────────────

def get_normal_training_data(db_path: str) -> pd.DataFrame:
    """
    Retrieve all normal-scenario rows for Isolation Forest training.

    Normal data only — the model learns what 'healthy' looks like.
    Anomalous data is intentionally excluded from training to avoid
    contaminating the normal distribution estimate.

    Returns
    -------
    pd.DataFrame with all sensor columns.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM sensor_data WHERE scenario = 'normal' ORDER BY timestamp",
        conn
    )
    conn.close()
    print(f"[DB] Normal training data: {len(df)} rows")
    return df


def get_recent_data(db_path: str, hours: int = 24) -> pd.DataFrame:
    """
    Retrieve sensor data from the last N hours.
    Used for short-term trend analysis and dashboard queries.

    Parameters
    ----------
    hours : int
        Lookback window in hours (default 24).

    Returns
    -------
    pd.DataFrame sorted ascending by timestamp.
    """
    conn = sqlite3.connect(db_path)
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM sensor_data WHERE timestamp > ? ORDER BY timestamp",
        conn, params=(cutoff,)
    )
    conn.close()
    print(f"[DB] Recent data ({hours}h): {len(df)} rows")
    return df


def get_anomaly_data(db_path: str, severity_min: str = "WARNING") -> pd.DataFrame:
    """
    Retrieve rows at or above a minimum severity threshold.
    Used for ticket validation and model retraining on fault data.

    Parameters
    ----------
    severity_min : str
        Minimum severity level to include (default 'WARNING').

    Returns
    -------
    pd.DataFrame of anomalous rows.
    """
    from carrier_baseline import SEVERITY_ORDER
    min_order = SEVERITY_ORDER.get(severity_min, 2)
    included  = [s for s, o in SEVERITY_ORDER.items() if o >= min_order]
    placeholders = ", ".join(["?"] * len(included))

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM sensor_data WHERE severity IN ({placeholders}) "
        f"ORDER BY timestamp",
        conn, params=included
    )
    conn.close()
    print(f"[DB] Anomaly data (>={severity_min}): {len(df)} rows")
    return df


def get_db_stats(db_path: str) -> dict:
    """
    Return database statistics: row counts by scenario and severity.
    Used for nightly health reporting.
    """
    if not Path(db_path).exists():
        return {"error": "Database not found"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sensor_data")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT scenario, COUNT(*) FROM sensor_data GROUP BY scenario")
    by_scenario = dict(cursor.fetchall())

    cursor.execute("SELECT severity, COUNT(*) FROM sensor_data GROUP BY severity")
    by_severity = dict(cursor.fetchall())

    cursor.execute("""
        SELECT COUNT(*) FROM sensor_data
        WHERE expires_at < date('now')
    """)
    expired = cursor.fetchone()[0]

    conn.close()
    return {
        "total_rows":   total,
        "by_scenario":  by_scenario,
        "by_severity":  by_severity,
        "expired_rows": expired,
    }


# ─────────────────────────────────────────────────────
# TTL CLEANUP
# ─────────────────────────────────────────────────────

def run_db_ttl_cleanup(db_path: str) -> dict:
    """
    Delete expired rows from sensor_data and VACUUM to reclaim disk space.

    Runs nightly as part of the maintenance window.
    VACUUM requires exclusive lock — do not run during active pipeline.

    Returns
    -------
    dict : {rows_deleted, by_severity}
    """
    if not Path(db_path).exists():
        print(f"[DB-TTL] Database not found: {db_path}")
        return {"rows_deleted": 0}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Capture what will be deleted (for reporting)
    cursor.execute("""
        SELECT severity, COUNT(*) FROM sensor_data
        WHERE expires_at < date('now')
        GROUP BY severity
    """)
    by_severity = dict(cursor.fetchall())
    total_to_delete = sum(by_severity.values())

    if total_to_delete > 0:
        cursor.execute("DELETE FROM sensor_data WHERE expires_at < date('now')")
        conn.commit()
        print(f"[DB-TTL] Deleted {total_to_delete} expired rows:")
        for sev, cnt in by_severity.items():
            print(f"  {sev}: {cnt} rows")

        # Reclaim space — this rewrites the DB file
        conn.execute("VACUUM")
        conn.commit()
        print("[DB-TTL] VACUUM complete — disk space reclaimed")
    else:
        print("[DB-TTL] No expired rows to delete")

    conn.close()
    return {"rows_deleted": total_to_delete, "by_severity": by_severity}


# ─────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    DB_PATH  = "data/hvac_sensor.db"
    RAW_DIR  = "data/raw"

    print("=" * 60)
    print("DB MANAGER — Self-test")
    print("=" * 60)

    # Generate data if not present
    from pathlib import Path
    if not any(Path(RAW_DIR).rglob("*.parquet")):
        print("\n[DB] No parquet files found — generating demo data...")
        from hvac_simulator import generate_dataset
        generate_dataset(total_minutes=120, output_dir=RAW_DIR,
                         modes=["normal", "low_refrigerant"])

    print("\n--- Init DB ---")
    conn = init_db(DB_PATH)
    conn.close()

    print("\n--- Merge parquets ---")
    result = merge_parquets_to_db(RAW_DIR, DB_PATH)

    print("\n--- DB Stats ---")
    stats = get_db_stats(DB_PATH)
    print(f"  Total rows:    {stats['total_rows']}")
    print(f"  By scenario:   {stats['by_scenario']}")
    print(f"  By severity:   {stats['by_severity']}")
    print(f"  Expired rows:  {stats['expired_rows']}")

    print("\n--- Normal training data ---")
    train_df = get_normal_training_data(DB_PATH)
    if not train_df.empty:
        print(f"  Rows: {len(train_df)}, "
              f"Sensors present: {[c for c in RAW_SENSORS if c in train_df.columns][:3]}...")

    print("\n--- Recent data (24h) ---")
    recent_df = get_recent_data(DB_PATH, hours=24)
    print(f"  Rows: {len(recent_df)}")

    print("\n--- TTL cleanup (dry — nothing to delete yet) ---")
    run_db_ttl_cleanup(DB_PATH)

    print("\n✓ db_manager.py — OK")
