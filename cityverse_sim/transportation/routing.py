"""
cityverse_sim.transportation.routing
====================================
Dynamic shortest-path routing, OD trip generation, and traffic assignment.
Supports passenger vehicles and emergency responder routing with dynamic flood re-routing.
"""

from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import numpy as np

from cityverse_sim.transportation.traffic_graph import DynamicRoadNetwork
from cityverse_sim.transportation.congestion import BPRCongestionModel


class TrafficRouter:
    """
    Manages dynamic path calculation, commuter traffic loading,
    and flood-induced detour assignment.
    """

    def __init__(
        self,
        dynamic_roads: DynamicRoadNetwork,
        congestion_model: Optional[BPRCongestionModel] = None,
    ):
        self.dyn_roads = dynamic_roads
        self.graph = dynamic_roads.graph
        self.congestion = congestion_model or BPRCongestionModel()
        
        # Background commuter demand OD matrix (Origin node, Destination node, trips/hr)
        self.od_trips: List[Tuple[str, str, float]] = []
        self._generate_baseline_od_trips()

    def _generate_baseline_od_trips(self, seed: int = 42):
        """Generates representative urban commuter demand between suburbs and urban core."""
        np.random.seed(seed)
        nodes = list(self.graph.nodes(data=True))
        
        # Identify suburbs (outer coords) and downtown (central core)
        suburb_nodes = [n for n, d in nodes if not (2100 <= d["x"] <= 4200 and 1800 <= d["y"] <= 3900)]
        core_nodes = [n for n, d in nodes if (2100 <= d["x"] <= 4200 and 1800 <= d["y"] <= 3900)]
        
        self.od_trips.clear()
        
        # Suburb -> Downtown commute trips
        for s in suburb_nodes[:30]:
            for c in core_nodes[:15]:
                flow = float(np.random.uniform(20.0, 80.0))
                self.od_trips.append((s, c, flow))
                
        # East-West transit cross-city trips
        west_nodes = [n for n, d in nodes if d["x"] < 1500]
        east_nodes = [n for n, d in nodes if d["x"] > 4500]
        for w in west_nodes[:10]:
            for e in east_nodes[:10]:
                flow = float(np.random.uniform(30.0, 100.0))
                self.od_trips.append((w, e, flow))

    def compute_shortest_path(
        self,
        origin: str,
        destination: str,
        vehicle_type: str = "passenger"
    ) -> Dict[str, Any]:
        """
        Finds the dynamic shortest path from origin to destination.
        
        Returns:
            Dict containing:
                is_reachable: bool
                path_nodes: List[str]
                path_edges: List[Tuple[str, str]]
                travel_time_s: float
                total_distance_m: float
        """
        if origin not in self.graph or destination not in self.graph:
            return {"is_reachable": False, "travel_time_s": float("inf"), "total_distance_m": 0.0, "path_nodes": []}
            
        if origin == destination:
            return {"is_reachable": True, "travel_time_s": 0.0, "total_distance_m": 0.0, "path_nodes": [origin]}

        # Build view or use weight attribute
        weight_attr = "dynamic_travel_time_s"
        
        try:
            path = nx.shortest_path(self.graph, source=origin, target=destination, weight=weight_attr)
            
            total_time_s = 0.0
            total_dist_m = 0.0
            edges = []
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edges.append((u, v))
                edata = self.graph[u][v]
                tt = edata.get(weight_attr, 10.0)
                if tt >= 1e7: # Encountered impassable link
                    return {"is_reachable": False, "travel_time_s": float("inf"), "total_distance_m": total_dist_m, "path_nodes": []}
                total_time_s += tt
                total_dist_m += edata.get("length_m", 0.0)
                
            return {
                "is_reachable": True,
                "path_nodes": path,
                "path_edges": edges,
                "travel_time_s": total_time_s,
                "total_distance_m": total_dist_m,
            }
        except nx.NetworkXNoPath:
            return {
                "is_reachable": False,
                "travel_time_s": float("inf"),
                "total_distance_m": 0.0,
                "path_nodes": [],
            }

    def assign_traffic_and_update(
        self,
        surface_water_m: np.ndarray,
        iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Coupled Traffic Assignment:
        1. Updates road water depths and flood speeds.
        2. Iteratively assigns OD flows to shortest paths and updates congested BPR travel times.
        3. Returns overall network performance statistics.
        """
        # Step 1: Update flood water depths on all edges
        self.dyn_roads.update_flood_conditions(surface_water_m)
        
        # Edge volume tracking: (u, v) -> total hourly volume
        edge_volumes: Dict[Tuple[str, str], float] = {(u, v): 0.0 for u, v in self.graph.edges()}
        
        # Step 2: Iterative Assignment (Method of Successive Averages)
        for it in range(iterations):
            # Compute travel times given current volumes
            self.congestion.update_network_travel_times(self.graph, edge_volumes, vehicle_type="passenger")
            
            # Temporary volume accumulator for this iteration
            iter_volumes: Dict[Tuple[str, str], float] = {(u, v): 0.0 for u, v in self.graph.edges()}
            
            for orig, dest, demand in self.od_trips:
                res = self.compute_shortest_path(orig, dest, vehicle_type="passenger")
                if res["is_reachable"]:
                    for edge in res["path_edges"]:
                        iter_volumes[edge] += demand
                        
            # Blend volumes
            step_size = 1.0 / (it + 1)
            for edge in edge_volumes:
                edge_volumes[edge] = (1.0 - step_size) * edge_volumes[edge] + step_size * iter_volumes[edge]
                
        # Final travel time update
        self.congestion.update_network_travel_times(self.graph, edge_volumes, vehicle_type="passenger")
        
        # Step 3: Compute aggregate metrics
        total_vkt = 0.0 # Vehicle-km traveled
        total_vht = 0.0 # Vehicle-hours traveled
        congested_edges = 0
        stranded_trips = 0
        total_trips = len(self.od_trips)
        
        for orig, dest, demand in self.od_trips:
            res = self.compute_shortest_path(orig, dest, vehicle_type="passenger")
            if res["is_reachable"]:
                total_vkt += demand * (res["total_distance_m"] / 1000.0)
                total_vht += demand * (res["travel_time_s"] / 3600.0)
            else:
                stranded_trips += 1
                
        for _, _, data in self.graph.edges(data=True):
            if data.get("vc_ratio", 0.0) > 1.0:
                congested_edges += 1
                
        avg_trip_time_min = (total_vht / max(1, total_trips - stranded_trips)) * 60.0
        
        road_summary = self.dyn_roads.get_summary_metrics()
        
        return {
            "closed_edges_passenger": road_summary["closed_edges_passenger"],
            "closed_edges_emergency": road_summary["closed_edges_emergency"],
            "passenger_closure_pct": road_summary["passenger_closure_pct"],
            "emergency_closure_pct": road_summary["emergency_closure_pct"],
            "congested_edges_count": congested_edges,
            "stranded_commute_trips": stranded_trips,
            "total_commute_trips": total_trips,
            "pct_trips_stranded": (stranded_trips / max(1, total_trips)) * 100.0,
            "avg_commute_time_min": avg_trip_time_min,
            "total_vehicle_hours_traveled": total_vht,
            "total_vehicle_km_traveled": total_vkt,
        }
