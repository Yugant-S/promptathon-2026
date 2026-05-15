"""
ttl_manager.py

Manages data retention (TTL) for parquet files and SQLite rows.

TTL policy rationale:
  - NORMAL data has no diagnostic value after 2 days
  - ANOMALY data must be retained for:
    (a) regulatory audit trails
    (b) model retraining datasets
    (c) post-repair validation window
  - TTL is NEVER reduced (only extended on anomaly confirmation)
  - All TTL state persists in a JSON log for crash recovery

Two-tier retention:
  Tier 1  Parquet files on disk (this module)
  Tier 2  SQLite rows (cleanup_sqlite_ttl helper)

"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from carrier_baseline import TTL_DAYS, SEVERITY_ORDER, compare_severity

# 
# CONSTANTS
# 

TTL_LOG_PATH = Path("data/ttl_log.json")

# 
# LOG I/O HELPERS
# 

def _load_log() -> dict:
    """Load TTL log from JSON. Returns empty dict if not found."""
    if TTL_LOG_PATH.exists():
        with open(TTL_LOG_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_log(log: dict):
    """Persist TTL log to JSON atomically.

    FIX-5: Old code did os.remove() then os.rename(), leaving a window where
    the log file didn't exist at all (crash/power-loss = permanent data loss).
    os.replace() is atomic on POSIX (rename(2) syscall).  On Windows it's
    best-effort within the same volume, which is the typical deployment case.
    We also fsync the temp file before the rename so bytes are durable.
    """
    TTL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TTL_LOG_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # ensure bytes reach disk before the rename
    os.replace(tmp, TTL_LOG_PATH)  # atomic on POSIX; best-effort on Windows


# 
# CORE TTL FUNCTIONS
# 

def register_file_ttl(filepath: str, worst_severity: str) -> dict:
    """
    Register a newly-written parquet file with its TTL.

    Called immediately after every batch flush from hvac_simulator.py
    and batch_pipeline.py. If the file is already registered (e.g.,
    pipeline restart), silently skips to avoid resetting an extended TTL.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the parquet file.
    worst_severity : str
        Highest severity label found in the batch.

    Returns
    -------
    dict : The registered TTL record.
    """
    log = _load_log()

    # Do not overwrite an existing record  it may have been extended
    if filepath in log:
        return log[filepath]

    ttl_days   = TTL_DAYS.get(worst_severity, TTL_DAYS["NORMAL"])
    created_at = datetime.now()
    expires_at = created_at + timedelta(days=ttl_days)

    record = {
        "created_at":      created_at.isoformat(),
        "worst_severity":  worst_severity,
        "ttl_days":        ttl_days,
        "expires_at":      expires_at.isoformat(),
        "extended":        False,
        "extended_at":     None,
    }

    log[filepath] = record
    _save_log(log)

    fname = Path(filepath).name
    print(f"[TTL] {fname}  severity={worst_severity}, "
          f"expires {expires_at.strftime('%Y-%m-%d')}")

    return record


def extend_ttl_on_anomaly(filepath: str, new_severity: str) -> dict:
    """
    Extend TTL when anomaly detection confirms a risk in a file.

    Monotonic rule: TTL can only increase (never decrease).
    This ensures anomalous data is not prematurely deleted if
    the initial batch appeared normal but ML confirms risk later.

    Parameters
    ----------
    filepath : str
        Path to the parquet file to extend.
    new_severity : str
        Confirmed anomaly severity from detection pipeline.

    Returns
    -------
    dict : Updated TTL record.
    """
    log = _load_log()

    if filepath not in log:
        # File was not registered  register it now
        return register_file_ttl(filepath, new_severity)

    record = log[filepath]
    old_severity   = record["worst_severity"]
    old_ttl_days   = record["ttl_days"]
    new_ttl_days   = TTL_DAYS.get(new_severity, old_ttl_days)

    # Only extend if new severity is higher than recorded
    effective_severity = compare_severity(old_severity, new_severity)
    effective_ttl      = TTL_DAYS.get(effective_severity, old_ttl_days)

    if effective_ttl <= old_ttl_days:
        # No extension needed
        return record

    created_at = datetime.fromisoformat(record["created_at"])
    new_expires = created_at + timedelta(days=effective_ttl)

    record["worst_severity"] = effective_severity
    record["ttl_days"]       = effective_ttl
    record["expires_at"]     = new_expires.isoformat()
    record["extended"]       = True
    record["extended_at"]    = datetime.now().isoformat()

    log[filepath] = record
    _save_log(log)

    fname = Path(filepath).name
    print(f"[TTL] Extended: {fname}")
    print(f"      {old_severity} ({old_ttl_days}d) -> "
          f"{effective_severity} ({effective_ttl}d), "
          f"new expiry {new_expires.strftime('%Y-%m-%d')}")

    return record


def run_ttl_cleanup(dry_run: bool = False) -> dict:
    """
    Scan all registered parquet files; delete expired ones.

    Parameters
    ----------
    dry_run : bool
        If True, report what would be deleted without deleting.

    Returns
    -------
    dict : Summary with deleted, kept, and orphaned file lists.
    """
    log = _load_log()
    now = datetime.now()

    deleted  = []
    kept     = []
    orphaned = []  # Registered but file no longer exists

    for filepath, record in list(log.items()):
        path = Path(filepath)

        if not path.exists():
            # File already gone (manually deleted or moved)
            orphaned.append(filepath)
            del log[filepath]
            continue

        expires_at = datetime.fromisoformat(record["expires_at"])

        if now > expires_at:
            # File has expired
            if not dry_run:
                path.unlink()
                del log[filepath]
            deleted.append({
                "file":     path.name,
                "severity": record["worst_severity"],
                "expired":  expires_at.strftime("%Y-%m-%d"),
            })
        else:
            days_remaining = (expires_at - now).days
            kept.append({
                "file":           path.name,
                "severity":       record["worst_severity"],
                "days_remaining": days_remaining,
                "expires":        expires_at.strftime("%Y-%m-%d"),
            })

    if not dry_run:
        _save_log(log)

    prefix = "[TTL-DRY]" if dry_run else "[TTL]"

    print(f"\n{prefix} Cleanup Summary:")
    if deleted:
        print(f"  Deleted ({len(deleted)}):")
        for d in deleted:
            print(f"     {d['file']} [{d['severity']}] expired {d['expired']}")
    else:
        print(f"  No expired files.")

    if kept:
        print(f"  Kept ({len(kept)}):")
        for k in kept:
            print(f"     {k['file']} [{k['severity']}] "
                  f"{k['days_remaining']}d remaining")

    if orphaned:
        print(f"  Orphaned (removed from log): {len(orphaned)}")

    return {"deleted": deleted, "kept": kept, "orphaned": orphaned}


def cleanup_sqlite_ttl(db_path: str) -> dict:
    """
    Delete expired rows from SQLite sensor_data table.

    Uses the expires_at column set at merge time.
    Runs VACUUM after deletion to reclaim disk space.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    dict : Rows deleted per severity level.
    """
    if not Path(db_path).exists():
        print(f"[TTL-DB] Database not found: {db_path}")
        return {}

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count by severity before deletion (for reporting)
    cursor.execute("""
        SELECT severity, COUNT(*) as cnt
        FROM sensor_data
        WHERE expires_at < date('now')
        GROUP BY severity
    """)
    rows_by_severity = dict(cursor.fetchall())

    if rows_by_severity:
        # Delete expired rows
        cursor.execute("""
            DELETE FROM sensor_data
            WHERE expires_at < date('now')
        """)
        deleted_count = cursor.rowcount
        conn.commit()

        # Reclaim disk space
        conn.execute("VACUUM")
        conn.commit()

        print(f"[TTL-DB] Deleted {deleted_count} expired rows:")
        for sev, cnt in rows_by_severity.items():
            print(f"  {sev}: {cnt} rows")
    else:
        print("[TTL-DB] No expired rows found.")

    conn.close()
    return rows_by_severity


def get_ttl_summary() -> dict:
    """
    Return current state of all registered files without modifying anything.
    Useful for dashboards and health checks.
    """
    log = _load_log()
    now = datetime.now()
    summary = {"total": len(log), "expired": 0, "active": 0, "by_severity": {}}

    for filepath, record in log.items():
        expires_at = datetime.fromisoformat(record["expires_at"])
        sev        = record["worst_severity"]
        summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
        if now > expires_at:
            summary["expired"] += 1
        else:
            summary["active"] += 1

    return summary


# 
# STANDALONE DEMO
# 
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("TTL MANAGER  Self-test")
    print("=" * 60)

    # Use temp file paths for demo (don't need real parquet files)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override log path for demo
        import ttl_manager as tm
        original_log_path = tm.TTL_LOG_PATH
        tm.TTL_LOG_PATH = Path(tmpdir) / "ttl_log.json"

        f1 = str(Path(tmpdir) / "normal_batch001.parquet")
        f2 = str(Path(tmpdir) / "low_refrigerant_batch001.parquet")
        f3 = str(Path(tmpdir) / "critical_batch001.parquet")

        # Create dummy files so cleanup finds them
        for f in [f1, f2, f3]:
            Path(f).touch()

        print("\n--- Registering files ---")
        tm.register_file_ttl(f1, "NORMAL")
        tm.register_file_ttl(f2, "WARNING")
        tm.register_file_ttl(f3, "ADVISORY")

        print("\n--- Extending TTL on anomaly ---")
        tm.extend_ttl_on_anomaly(f2, "CRITICAL")

        print("\n--- TTL Summary ---")
        summary = tm.get_ttl_summary()
        print(f"  Total registered: {summary['total']}")
        print(f"  Active: {summary['active']}, Expired: {summary['expired']}")
        print(f"  By severity: {summary['by_severity']}")

        print("\n--- Dry-run cleanup ---")
        tm.run_ttl_cleanup(dry_run=True)

        tm.TTL_LOG_PATH = original_log_path

    print("\n ttl_manager.py  OK")
