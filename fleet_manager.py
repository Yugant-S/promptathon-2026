"""
fleet_manager.py

Fleet-wide prioritization and orchestration.
Aggregates maintenance tickets across all Carrier units
and calculates a dynamic priority score based on:
1. Financial Loss (Cost/Day)
2. Time Urgency (Hours to Failure)
3. Operational Impact (Zone Criticality)

"""

import os
import json
from datetime import datetime
from pathlib import Path

# Mapping zone type to operational importance (1-10)
ZONE_CRITICALITY_SCORE = {
    "Production Floor": 10,
    "Cold Storage":      8,
    "Clean Room":       10,
    "Server Room":       7,
    "Warehouse":         5,
    "Office":            3,
    "Storage Room":      2
}

class FleetManager:
    """
    Manages a fleet of HVAC units, prioritizing maintenance actions.
    """

    def __init__(self):
        self.fleet_status = {}
        self.priority_queue = []
        
        # Load FLEET_CONFIG from hvac_simulator
        try:
            from hvac_simulator import FLEET_CONFIG
            self.fleet_config = FLEET_CONFIG
        except ImportError:
            self.fleet_config = {}

    def update_unit(self, unit_id: str, ticket: dict) -> None:
        """
        Update a unit's status in the fleet based on a new ticket.
        """
        # Extract from ticket (Updated structure)
        severity = ticket.get("severity", "NORMAL")
        freeze_risk_pct = ticket.get("freeze_risk_pct", 0)
        
        urgency_obj = ticket.get("urgency", {})
        urgency_hours = urgency_obj.get("hours", 672)
        urgency_display = urgency_obj.get("display", "~4 weeks")
        
        dominant_cause = ticket.get("dominant_cause", "normal")
        
        # Extract cost escalation
        cost_escalation = ticket.get("cost_escalation", {})
        cost_per_day_inr = cost_escalation.get("cost_per_day_delay_inr", 0)
        
        # Get zone info
        config = self.fleet_config.get(unit_id, {"zone": "Unknown", "criticality": "MEDIUM"})
        zone = config["zone"]
        
        # Determine zone criticality score
        zone_score = 5 # Default
        for zone_type, score in ZONE_CRITICALITY_SCORE.items():
            if zone_type in zone:
                zone_score = score
                break
        
        # Store status
        self.fleet_status[unit_id] = {
            "severity": severity,
            "freeze_risk_pct": freeze_risk_pct,
            "urgency_hours": urgency_hours,
            "urgency_display": urgency_display,
            "dominant_cause": dominant_cause,
            "cost_per_day_inr": cost_per_day_inr,
            "zone": zone,
            "zone_criticality_score": zone_score,
            "ticket_id": ticket.get("ticket_id"),
            "last_updated": datetime.now()
        }
        
        # Recalculate priority
        self.priority_queue = self.get_dispatch_queue()

    def calculate_priority_score(self, unit_id: str) -> float:
        """
        Dynamic priority scoring formula for Carrier HVAC units.
        """
        status = self.fleet_status[unit_id]
        
        # cost_score 35%: financial urgency
        cost_score = min(status["cost_per_day_inr"] / 100, 100)
        
        # urgency_score 40%: time criticality
        urgency_score = 100 / max(status["urgency_hours"], 1)
        
        # zone_score 25%: operational impact
        zone_score = status["zone_criticality_score"] * 10
        
        # severity_multiplier: exponential weight for serious faults
        severity_multiplier = {
            "NORMAL": 0, 
            "ADVISORY": 1,
            "WARNING": 2, 
            "CRITICAL": 4,
            "IMMINENT_FAILURE": 8
        }.get(status["severity"], 0)
        
        score = (
            cost_score    * 0.35 +
            urgency_score * 0.40 +
            zone_score    * 0.25
        ) * severity_multiplier
        
        return round(score, 2)

    def get_dispatch_queue(self) -> list:
        """
        Returns list of dicts sorted by priority_score DESC.
        """
        queue = []
        for unit_id in self.fleet_status:
            score = self.calculate_priority_score(unit_id)
            status = self.fleet_status[unit_id]
            
            action = "NO ACTION NEEDED"
            if score > 50: action = "DISPATCH IMMEDIATELY"
            elif score > 20: action = "SCHEDULE THIS WEEK"
            elif score > 0: action = "MONITOR CLOSELY"
            
            # Special case for sensor faults (if ticket_id starts with SENSOR)
            if status.get("ticket_id", "").startswith("SENSOR"):
                action = "REPLACE SENSOR"
            
            queue.append({
                "unit_id": unit_id,
                "zone": status["zone"],
                "severity": status["severity"],
                "urgency_hours": status["urgency_hours"],
                "urgency_display": status["urgency_display"],
                "dominant_cause": status["dominant_cause"],
                "priority_score": score,
                "cost_per_day_inr": status["cost_per_day_inr"],
                "action": action
            })
            
        # Sort by priority score
        queue.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Add rank
        for i, item in enumerate(queue):
            item["rank"] = i + 1
            
        return queue

    def get_fleet_summary(self) -> dict:
        """
        Aggregated fleet health and financial loss metrics.
        """
        severities = {"NORMAL": 0, "ADVISORY": 0, "WARNING": 0, "CRITICAL": 0, "IMMINENT_FAILURE": 0}
        total_loss = 0
        risk_weighted_sum = 0
        total_units = len(self.fleet_status)
        
        if total_units == 0:
            return {"total_units": 0, "fleet_health_score": 100}

        for unit_id, status in self.fleet_status.items():
            severities[status["severity"]] = severities.get(status["severity"], 0) + 1
            total_loss += status["cost_per_day_inr"]
            
            # Weighted risk for health score
            sev_weight = {"NORMAL": 0, "ADVISORY": 10, "WARNING": 30, "CRITICAL": 60, "IMMINENT_FAILURE": 100}
            risk_weighted_sum += sev_weight.get(status["severity"], 0)

        health_score = max(0, 100 - (risk_weighted_sum / total_units))
        
        queue = self.get_dispatch_queue()
        highest_priority = queue[0]["unit_id"] if queue else None

        return {
            "total_units": total_units,
            "units_by_severity": severities,
            "total_daily_loss_inr": total_loss,
            "highest_priority_unit": highest_priority,
            "fleet_health_score": round(health_score, 1)
        }

    def print_fleet_dashboard(self) -> None:
        """
        Prints the production-grade fleet status dashboard.
        """
        queue = self.get_dispatch_queue()
        summary = self.get_fleet_summary()
        
        print("\n+--------------------------------------------------------------+")
        print(f"|           CARRIER FLEET STATUS - {summary['total_units']} Units                  |")
        print("+------+------------------+----------+------------+------------+")
        print("| Rank | Unit ID          | Severity | Lead Time  | Cost/Day   |")
        print("+------+------------------+----------+------------+------------+")
        
        for item in queue[:10]: # Show top 10
            print(f"|  {item['rank']:<3} | {item['unit_id']:<16} | {item['severity']:<8} | {item['urgency_display']:>10} | Rs {item['cost_per_day_inr']:<9,} |")
            
        print("+--------------------------------------------------------------+")
        print(f"Fleet Health Score: {summary['fleet_health_score']}/100")
        print(f"Total Daily Loss:   Rs {summary['total_daily_loss_inr']:,}")
        
        attention_count = sum(1 for item in queue if item["priority_score"] > 0)
        print(f"Units Needing Attention: {attention_count}")

    def generate_savings_report(self) -> dict:
        """
        Calculates maintenance savings from early detection.
        """
        # Business constants from instructions
        EMERGENCY_COST = 74000
        
        events_caught_early = 0
        total_actual_cost = 0
        
        # Scan alerts/tickets directory
        ticket_dir = Path("alerts/tickets")
        if not ticket_dir.exists():
            return {"error": "No tickets found"}
            
        # Dedup tickets: only count the latest ticket per unit to avoid overcounting costs
        latest_tickets = {}
        for tf in ticket_dir.glob("*.json"):
            with open(tf) as f:
                t = json.load(f)
            
            unit_id = t.get("unit_id", "unknown")
            gen_at = t.get("generated_at", "")
            
            if unit_id not in latest_tickets or gen_at > latest_tickets[unit_id].get("generated_at", ""):
                latest_tickets[unit_id] = t

        for t in latest_tickets.values():
            # Logic: If detected at ADVISORY or WARNING, it's early
            # (Note: In a real system we'd track the progression)
            if t.get("severity") in ["ADVISORY", "WARNING"]:
                events_caught_early += 1
                
            # Actual cost from ticket cost escalation
            cost_now = t.get("cost_escalation", {}).get("repair_cost_now_inr", 25000)
            total_actual_cost += cost_now

        emergency_cost_avoided = events_caught_early * EMERGENCY_COST
        net_savings = emergency_cost_avoided - total_actual_cost
        
        report = {
            "events_detected_early": events_caught_early,
            "emergency_cost_avoided": emergency_cost_avoided,
            "actual_maintenance_cost": total_actual_cost,
            "net_savings": net_savings
        }
        
        print(f"\nEvents detected early: {events_caught_early}")
        print(f"Emergency cost avoided: Rs {emergency_cost_avoided:,}")
        print(f"Actual maintenance cost: Rs {total_actual_cost:,}")
        print(f"Net savings: Rs {net_savings:,}")
        
        return report

if __name__ == "__main__":
    fm = FleetManager()
    # Dummy update for self-test
    fm.update_unit("CARRIER-30XA-01", {
        "ticket_id": "TKT-001",
        "severity": "CRITICAL",
        "total_freeze_risk": 85,
        "urgency_hours": 6,
        "dominant_cause": "low_refrigerant",
        "cost_escalation": {"cost_per_day_delay_inr": 4200, "repair_cost_now_inr": 12000}
    })
    fm.print_fleet_dashboard()
    fm.generate_savings_report()
