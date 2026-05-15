"""
hvac_simulator.py

Physics-informed Carrier 30XA HVAC sensor data simulator.
Generates 1 row/minute with realistic fault progression.
Now upgraded with Fleet Management and Defrost Cycle Suppression.

Design principles:
  - Gaussian baseline noise calibrated to Carrier 30XA std devs
  - Fault drifts follow thermodynamic causality chains
  - Defrost cycle simulation for false-positive suppression testing
  - Fleet-wide multi-unit generation support

"""

import os
import time
import math
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from carrier_baseline import (
    CARRIER_BASELINE, FREEZE_CAUSE_WEIGHTS,
    TTL_DAYS, get_z_score, get_severity_from_z, SEVERITY_ORDER,
    calculate_freeze_risk,  # FIX-1: canonical copy, removed duplicate below
)

# 
# FLEET CONFIGURATION
# 

FLEET_CONFIG = {
    "CARRIER-30XA-01": {
        "zone": "Production Floor A",
        "scenario": "low_refrigerant",
        "anomaly_start_min": 60,
        "criticality": "HIGH"
    },
    "CARRIER-30XA-02": {
        "zone": "Cold Storage B",
        "scenario": "dirty_coil",
        "anomaly_start_min": 45,
        "criticality": "HIGH"
    },
    "CARRIER-30XA-03": {
        "zone": "Office Block",
        "scenario": "normal",
        "anomaly_start_min": 9999,
        "criticality": "LOW"
    },
    "CARRIER-30XA-04": {
        "zone": "Production Floor B",
        "scenario": "exv_malfunction",
        "anomaly_start_min": 30,
        "criticality": "HIGH"
    },
    "CARRIER-30XA-05": {
        "zone": "Warehouse",
        "scenario": "low_chilled_water",
        "anomaly_start_min": 50,
        "criticality": "MEDIUM"
    },
    "CARRIER-30XA-06": {
        "zone": "Clean Room",
        "scenario": "condenser_issue",
        "anomaly_start_min": 70,
        "criticality": "HIGH"
    },
    "CARRIER-30XA-07": {
        "zone": "Server Room",
        "scenario": "normal",
        "anomaly_start_min": 9999,
        "criticality": "MEDIUM"
    },
    "CARRIER-30XA-08": {
        "zone": "Production Floor C",
        "scenario": "multi_fault",
        "anomaly_start_min": 40,
        "criticality": "HIGH"
    },
    "CARRIER-30XA-09": {
        "zone": "Storage Room",
        "scenario": "normal",
        "anomaly_start_min": 9999,
        "criticality": "LOW"
    },
    "CARRIER-30XA-10": {
        "zone": "Cold Storage A",
        "scenario": "low_refrigerant",
        "anomaly_start_min": 80,
        "criticality": "HIGH"
    }
}

ALL_SCENARIOS = [
    "normal", "low_refrigerant", "dirty_coil",
    "exv_malfunction", "low_chilled_water",
    "condenser_issue", "multi_fault",
]

# FIX-1: calculate_freeze_risk() moved to carrier_baseline.py
# Import it from there; this copy is removed to prevent silent divergence.


class HVACSimulator:
    """
    Stateful simulator for a single Carrier HVAC unit.
    Handles fault progression and defrost cycles.
    """
    
    def __init__(self, mode: str = "normal", seed: int = 42, stuck_sensor: str = None):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.stuck_sensor = stuck_sensor
        self.stuck_value = None
        
        # Defrost state
        self.next_defrost_at = random.randint(240, 360) # 4-6 hours
        self.defrost_duration = 15
        self.in_defrost = False
        self.defrost_end_at = None

    def _baseline_noise(self, sensor: str) -> float:
        b = CARRIER_BASELINE[sensor]
        return float(self.rng.normal(b["mean"], b["std"]))

    def _next_row(self, timestamp: datetime) -> dict:
        """Generate the next minute of sensor data."""
        #  Baseline readings
        row = {s: self._baseline_noise(s) for s in CARRIER_BASELINE}
        
        #  Apply scenario-specific fault progressions
        onset = 60 # Default onset for most faults
        if self.mode == "dirty_coil": onset = 45
        elif self.mode == "exv_malfunction": onset = 30
        elif self.mode == "low_chilled_water": onset = 50
        elif self.mode == "condenser_issue": onset = 70
        elif self.mode == "multi_fault": onset = 40

        if self.mode != "normal" and self.step > onset:
            t = self.step - onset
            if self.mode == "low_refrigerant":
                row["suction_pressure_bar"] -= 0.005 * t + self.rng.normal(0, 0.005)
                row["coil_delta_T_C"]       -= 0.004 * t + self.rng.normal(0, 0.002)
                row["subcooling_C"]         -= 0.006 * t + self.rng.normal(0, 0.003)
                row["superheat_C"]          += 0.003 * t + self.rng.normal(0, 0.005)
            elif self.mode == "dirty_coil":
                row["coil_delta_T_C"]    -= 0.008 * t + self.rng.normal(0, 0.005)
                row["blower_current_A"]  += 0.005 * t + self.rng.normal(0, 0.002)
                row["supply_air_temp_C"] += 0.006 * t + self.rng.normal(0, 0.003)
            elif self.mode == "exv_malfunction":
                row["superheat_C"] += 3.0 * math.sin(2 * math.pi * t / 30)
                row["suction_pressure_bar"] -= 0.002 * t * 0.1 + self.rng.normal(0, 0.01)
            elif self.mode == "low_chilled_water":
                row["chilled_water_flow_LPS"]    -= 0.05 * t + self.rng.normal(0, 0.02)
                row["chilled_water_delta_T_C"]   += 0.008 * t + self.rng.normal(0, 0.005)
            elif self.mode == "condenser_issue":
                row["discharge_pressure_bar"]   += 0.008 * t + self.rng.normal(0, 0.005)
                row["compressor_current_A"]     += 0.04 * t + self.rng.normal(0, 0.01)
                row["compressor_runtime_pct"]   += 0.03 * t + self.rng.normal(0, 0.01)
            elif self.mode == "multi_fault":
                f = 0.5
                row["suction_pressure_bar"] -= f * 0.01 * t + self.rng.normal(0, 0.005)
                row["coil_delta_T_C"]       -= f * 0.01 * t + self.rng.normal(0, 0.005)
                row["subcooling_C"]         -= f * 0.01 * t + self.rng.normal(0, 0.005)
                row["blower_current_A"]     += f * 0.01 * t + self.rng.normal(0, 0.005)

        #  DEFROST CYCLE LOGIC
        if self.step == self.next_defrost_at:
            self.in_defrost = True
            self.defrost_end_at = self.step + self.defrost_duration
            self.next_defrost_at = self.step + random.randint(240, 360)
        
        if self.in_defrost and self.step >= self.defrost_end_at:
            self.in_defrost = False
            
        if self.in_defrost:
            # Coil temp spikes, supply air warms, blower slows
            row["coil_delta_T_C"] += 4.0
            row["supply_air_temp_C"] += 3.0
            row["blower_current_A"] -= 2.0
            row["blower_current_A"] = max(row["blower_current_A"], 2.0)
            row["is_defrost_cycle"] = 1
            row["defrost_recovery"] = 0
        elif self.defrost_end_at and (0 < self.step - self.defrost_end_at <= 20):
            row["is_defrost_cycle"] = 0
            row["defrost_recovery"] = 1
        else:
            row["is_defrost_cycle"] = 0
            row["defrost_recovery"] = 0

        # ── Apply stuck sensor fault
        if self.stuck_sensor and self.stuck_sensor in row:
            if self.stuck_value is None:
                self.stuck_value = row[self.stuck_sensor]
            row[self.stuck_sensor] = self.stuck_value

        # ── Physical Clamping
        row["suction_pressure_bar"]    = max(1.0, row["suction_pressure_bar"])
        row["discharge_pressure_bar"]  = max(row["suction_pressure_bar"] + 1.0, row["discharge_pressure_bar"])
        row["chilled_water_flow_LPS"]  = max(10.0, row["chilled_water_flow_LPS"])
        row["superheat_C"]             = max(0.0, row["superheat_C"])
        row["subcooling_C"]            = max(0.0, row["subcooling_C"])
        row["compressor_runtime_pct"]  = min(100.0, max(0.0, row["compressor_runtime_pct"]))

        #  Derived Sensors
        row["compression_ratio"]          = round(row["discharge_pressure_bar"] / row["suction_pressure_bar"], 4)
        row["coil_efficiency_index"]       = round(row["coil_delta_T_C"] / (row["blower_current_A"] * 0.1), 4)
        # FIX-6: corrected COP formula.
        # Q_cooling (kW) = cp_water * flow_LPS * delta_T  (4.186 kJ/kg/°C, density ~1 kg/L)
        # P_compressor (kW) = V * I * pf / 1000  (415 V line, pf 0.85 ⇒ factor 0.353)
        _q_kw = 4.186 * row["chilled_water_flow_LPS"] * row["chilled_water_delta_T_C"]
        _p_kw = row["compressor_current_A"] * 0.353  # 415 V * 0.85 pf / 1000
        row["system_COP"] = round(_q_kw / (_p_kw + 1e-6), 4)

        #  Risk Assessment
        risk = calculate_freeze_risk(row)
        row.update(risk)
        
        #  Metadata
        row["timestamp"] = timestamp.isoformat()
        row["scenario"]  = self.mode
        row["minute"]    = self.step
        
        self.step += 1
        return row

# Backward compatibility for existing code
def generate_row(scenario, minute, rng, timestamp):
    sim = HVACSimulator(mode=scenario)
    sim.rng = rng
    sim.step = minute
    return sim._next_row(timestamp)

# 
# BATCH WRITER
# 

class TTLAwareBatchWriter:
    FLUSH_SIZE = 60
    def __init__(self, unit_id: str, output_dir: str):
        self.unit_id = unit_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.batch_count = 0

    def append(self, row: dict):
        self.buffer.append(row)
        if len(self.buffer) >= self.FLUSH_SIZE:
            self.flush()

    def flush(self):
        if not self.buffer: return
        df = pd.DataFrame(self.buffer)
        self.batch_count += 1
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.unit_id}_batch{self.batch_count:03d}_{ts_str}.parquet"
        filepath = self.output_dir / filename
        df.to_parquet(filepath, index=False)
        
        # Determine severity for TTL
        worst_sev = "NORMAL"
        if "freeze_severity" in df.columns:
            worst_sev = df["freeze_severity"].iloc[df["freeze_severity"].map(SEVERITY_ORDER).argmax()]
            
        try:
            from ttl_manager import register_file_ttl
            register_file_ttl(str(filepath), worst_sev)
        except Exception as e:  # FIX-3: was bare except:pass, swallowed all errors
            logging.warning("TTL registration failed for %s: %s", filepath, e)
        self.buffer.clear()

    def close(self):
        self.flush()

# 
# GENERATORS
# 

def generate_fleet_data(total_minutes=480, output_dir="data/raw"):
    """Generate multi-unit fleet data based on FLEET_CONFIG."""
    print(f"[FLEET] Generating data for {len(FLEET_CONFIG)} units...")
    start_ts = datetime.now()
    results = {}
    
    summary_data = []
    
    for unit_id, config in FLEET_CONFIG.items():
        print(f"[UNIT] {unit_id} | {config['zone']} | {config['scenario']}")
        
        # Inject sensor fault for only ONE unit to show the feature
        stuck = "blower_current_A" if unit_id == "CARRIER-30XA-08" else None
        sim = HVACSimulator(mode=config["scenario"], stuck_sensor=stuck)
        # Group by unit_id then scenario
        writer = TTLAwareBatchWriter(unit_id, Path(output_dir) / unit_id / config["scenario"])
        
        for minute in range(total_minutes):
            row = sim._next_row(start_ts + timedelta(minutes=minute))
            row["unit_id"] = unit_id
            writer.append(row)
            
        writer.close()
        
        anon_at = f"min {config['anomaly_start_min']}" if config['anomaly_start_min'] < 9999 else "N/A"
        summary_data.append([unit_id, config["zone"], config["scenario"], anon_at])

    # Print summary table
    print("\nUnit ID          | Zone              | Scenario        | Anomaly At")
    print("-" * 75)
    for row in summary_data:
        print(f"{row[0]:<16} | {row[1]:<17} | {row[2]:<15} | {row[3]}")
    print()

def generate_dataset(total_minutes=480, output_dir="data/raw", modes=None, seed=42):
    """Original generator for backward compatibility."""
    scenarios = modes if modes else ["normal", "low_refrigerant", "dirty_coil", "exv_malfunction", "low_chilled_water", "condenser_issue", "multi_fault"]
    start_ts = datetime(2025, 7, 1, 6, 0, 0)
    
    for scenario in scenarios:
        sim = HVACSimulator(mode=scenario, seed=seed)
        writer = TTLAwareBatchWriter(scenario, Path(output_dir) / scenario)
        for minute in range(total_minutes):
            row = sim._next_row(start_ts + timedelta(minutes=minute))
            writer.append(row)
        writer.close()
    return {}

if __name__ == "__main__":
    generate_fleet_data(total_minutes=120)
    print("\nOK hvac_simulator.py - OK")
