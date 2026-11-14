"""
cityverse_sim.emergency.dispatch
================================
Emergency medical services (EMS) dispatch simulator and trauma hospital routing.
Models ambulance dispatch from EMS bases, patient rescue, hospital transport,
and emergency accessibility decay under severe flooding.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from cityverse_sim.city.network import FacilityRegistry
from cityverse_sim.transportation.traffic_graph import DynamicRoadNetwork
from cityverse_sim.transportation.routing import TrafficRouter


class EmergencyDispatcher:
    """
    Manages ambulance fleets, emergency call routing, and hospital accessibility.
    """

    def __init__(
        self,
        facilities: FacilityRegistry,
        router: TrafficRouter,
    ):
        self.facilities = facilities
        self.router = router
        self.graph = router.graph
        
        # Cache facility road nodes
        self.ems_bases = self.facilities.get_by_category("ems_stations")
        self.hospitals = self.facilities.get_by_category("hospitals")
        
        # Active dispatch logs
        self.dispatches: List[Dict[str, Any]] = []

    def dispatch_call(
        self,
        call_id: str,
        target_node: str,
        priority: int = 1
    ) -> Dict[str, Any]:
        """
        Dispatches the closest available ambulance to target_node,
        then routes from target_node to the closest accessible Level-1 Trauma Hospital.
        
        Returns:
            Dict containing dispatch metrics (EMS travel time, hospital travel time, total time, status).
        """
        # Step 1: Find closest reachable EMS base
        best_ems = None
        min_ems_time = float("inf")
        best_ems_res = None
        
        for ems in self.ems_bases:
            ems_node = ems["nearest_road_node"]
            res = self.router.compute_shortest_path(ems_node, target_node, vehicle_type="emergency")
            if res["is_reachable"] and res["travel_time_s"] < min_ems_time:
                min_ems_time = res["travel_time_s"]
                best_ems = ems
                best_ems_res = res
                
        if best_ems is None:
            # Target is completely cut off from all EMS bases
            record = {
                "call_id": call_id,
                "target_node": target_node,
                "status": "UNREACHABLE_ISOLATED",
                "ems_id": None,
                "hospital_id": None,
                "ems_travel_time_min": float("inf"),
                "hospital_travel_time_min": float("inf"),
                "total_response_time_min": float("inf"),
                "golden_hour_exceeded": True,
            }
            self.dispatches.append(record)
            return record

        # Step 2: Route patient from target_node to closest reachable Trauma Hospital
        best_hosp = None
        min_hosp_time = float("inf")
        best_hosp_res = None
        
        for hosp in self.hospitals:
            hosp_node = hosp["nearest_road_node"]
            res = self.router.compute_shortest_path(target_node, hosp_node, vehicle_type="emergency")
            if res["is_reachable"] and res["travel_time_s"] < min_hosp_time:
                min_hosp_time = res["travel_time_s"]
                best_hosp = hosp
                best_hosp_res = res
                
        if best_hosp is None:
            # Patient reached by EMS, but all hospitals cut off
            record = {
                "call_id": call_id,
                "target_node": target_node,
                "status": "HOSPITAL_UNREACHABLE",
                "ems_id": best_ems["id"],
                "hospital_id": None,
                "ems_travel_time_min": min_ems_time / 60.0,
                "hospital_travel_time_min": float("inf"),
                "total_response_time_min": float("inf"),
                "golden_hour_exceeded": True,
            }
            self.dispatches.append(record)
            return record

        total_time_min = (min_ems_time + min_hosp_time) / 60.0
        record = {
            "call_id": call_id,
            "target_node": target_node,
            "status": "SUCCESS",
            "ems_id": best_ems["id"],
            "hospital_id": best_hosp["id"],
            "ems_travel_time_min": min_ems_time / 60.0,
            "hospital_travel_time_min": min_hosp_time / 60.0,
            "total_response_time_min": total_time_min,
            "golden_hour_exceeded": bool(total_time_min > 30.0),
        }
        self.dispatches.append(record)
        return record

    def evaluate_emergency_coverage(self, sample_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Assesses city-wide emergency coverage by running simulated dispatches
        across a representative distribution of city intersections.
        """
        if sample_nodes is None:
            sample_nodes = list(self.graph.nodes())
            
        total_calls = len(sample_nodes)
        successful_calls = 0
        stranded_calls = 0
        golden_hour_violations = 0
        total_response_time = 0.0
        
        for idx, node in enumerate(sample_nodes):
            res = self.dispatch_call(f"CALL-{idx:04d}", node)
            if res["status"] == "SUCCESS":
                successful_calls += 1
                total_response_time += res["total_response_time_min"]
                if res["golden_hour_exceeded"]:
                    golden_hour_violations += 1
            else:
                stranded_calls += 1
                golden_hour_violations += 1
                
        avg_response_time = (total_response_time / max(1, successful_calls))
        coverage_pct = (successful_calls / max(1, total_calls)) * 100.0
        
        return {
            "total_assessed_nodes": total_calls,
            "reachable_nodes": successful_calls,
            "isolated_stranded_nodes": stranded_calls,
            "coverage_pct": coverage_pct,
            "avg_response_time_min": avg_response_time,
            "golden_hour_violation_pct": (golden_hour_violations / max(1, total_calls)) * 100.0,
        }

