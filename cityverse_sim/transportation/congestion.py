"""
cityverse_sim.transportation.congestion
=======================================
Volume-delay functions (Bureau of Public Roads - BPR) combining
flood-induced speed reduction with dynamic vehicle congestion delays.
"""

from typing import Dict, Any, Tuple
import networkx as nx
import numpy as np


class BPRCongestionModel:
    """
    Computes dynamic edge travel times combining flood-induced speed limits
    and traffic volume congestion via the Bureau of Public Roads (BPR) function:
        T_e(V_e, d_e) = (L_e / v_flood) * [ 1 + alpha * (V_e / C_e)^beta ]
    """

    def __init__(
        self,
        alpha_bpr: float = 0.15,
        beta_bpr: float = 4.0,
        impassable_penalty_s: float = 1e8,
    ):
        self.alpha = float(alpha_bpr)
        self.beta = float(beta_bpr)
        self.impassable_penalty = float(impassable_penalty_s)

    def compute_edge_travel_time(
        self,
        length_m: float,
        flood_speed_mps: float,
        volume_veh_hr: float,
        capacity_veh_hr: float,
        is_passable: bool = True
    ) -> float:
        """
        Calculates travel time for an individual edge in seconds.
        """
        if not is_passable or flood_speed_mps <= 0.05:
            return self.impassable_penalty
            
        # Free-flow travel time under current weather/flood condition
        t0_s = length_m / flood_speed_mps
        
        # BPR congestion multiplier
        vc_ratio = volume_veh_hr / max(capacity_veh_hr, 1.0)
        delay_factor = 1.0 + self.alpha * (vc_ratio**self.beta)
        
        return float(t0_s * delay_factor)

    def update_network_travel_times(
        self,
        graph: nx.DiGraph,
        edge_volumes: Dict[Tuple[str, str], float],
        vehicle_type: str = "passenger" # "passenger" or "emergency"
    ):
        """
        Updates 'dynamic_travel_time_s' on all edges in graph in-place.
        """
        for u, v, data in graph.edges(data=True):
            vol = edge_volumes.get((u, v), 0.0)
            cap = data.get("capacity_veh_per_hr", 1000.0)
            length = data.get("length_m", 100.0)
            flood_speed = data.get("flood_speed_mps", data.get("free_flow_speed_mps", 15.0))
            
            if vehicle_type == "emergency":
                is_pass = data.get("is_passable_emergency", True)
            else:
                is_pass = data.get("is_passable_passenger", True)
                
            tt_s = self.compute_edge_travel_time(
                length_m=length,
                flood_speed_mps=flood_speed,
                volume_veh_hr=vol,
                capacity_veh_hr=cap,
                is_passable=is_pass
            )
            
            data["dynamic_travel_time_s"] = tt_s
            data["current_volume_veh_hr"] = vol
            data["vc_ratio"] = vol / max(cap, 1.0)

