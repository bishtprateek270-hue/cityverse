"""
cityverse_sim.transportation.traffic_graph
==========================================
Bridges the surface inundation grid h(x, y, t) with the vector road network.
Samples water depths along road geometries, updates edge passability,
and computes flood-induced speed degradation using empirical urban curves.
"""

from typing import Dict, Any, List, Tuple
import networkx as nx
import numpy as np

from cityverse_sim.city.grid import CityGrid
from cityverse_sim.city.network import RoadNetwork


class DynamicRoadNetwork:
    """
    Augments RoadNetwork with spatial water depth sampling,
    Pregnolato speed reduction curves, and dynamic passability tagging.
    """

    def __init__(
        self,
        road_net: RoadNetwork,
        grid: CityGrid,
        crit_depth_passenger_m: float = 0.15,
        crit_depth_emergency_m: float = 0.30,
    ):
        self.road_net = road_net
        self.graph = road_net.graph
        self.grid = grid
        self.crit_depth_passenger = float(crit_depth_passenger_m)
        self.crit_depth_emergency = float(crit_depth_emergency_m)
        
        # Precompute sampling points for every directed edge
        self.edge_sample_cells: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        self._precompute_edge_cells()

    def _precompute_edge_cells(self, sample_spacing_m: float = 25.0):
        """Samples road polyline coordinates at regular intervals."""
        for u, v, data in self.graph.edges(data=True):
            u_node = self.graph.nodes[u]
            v_node = self.graph.nodes[v]
            
            x1, y1 = u_node["x"], u_node["y"]
            x2, y2 = v_node["x"], v_node["y"]
            length = data.get("length_m", np.hypot(x2 - x1, y2 - y1))
            
            n_samples = max(2, int(np.ceil(length / sample_spacing_m)))
            xs = np.linspace(x1, x2, n_samples)
            ys = np.linspace(y1, y2, n_samples)
            
            cells = []
            seen = set()
            for x, y in zip(xs, ys):
                r, c = self.grid.coord_to_cell(x, y)
                if (r, c) not in seen:
                    seen.add((r, c))
                    cells.append((r, c))
                    
            self.edge_sample_cells[(u, v)] = cells

    def update_flood_conditions(self, surface_water_m: np.ndarray):
        """
        Samples surface water depth for every road edge and recalculates
        effective travel speeds and accessibility status.
        
        Args:
            surface_water_m: (H, W) array of surface inundation depths [m]
        """
        for (u, v), cells in self.edge_sample_cells.items():
            edge_data = self.graph[u][v]
            is_bridge = edge_data.get("is_bridge", False)
            
            if is_bridge:
                # Bridges only submerge if the water level in the river channel exceeds the deck
                # Find water surface elevation at river cells beneath the bridge
                deck_elev = max(self.graph.nodes[u]["z"], self.graph.nodes[v]["z"]) + 1.5
                edge_data["elevation_m"] = deck_elev
                
                river_cells = [c for c in cells if self.grid.river_mask[c[0], c[1]]]
                if river_cells:
                    max_water_elev = max(self.grid.dem[r, c] + surface_water_m[r, c] for r, c in river_cells)
                    effective_depth = max(0.0, float(max_water_elev - deck_elev))
                else:
                    # Bridge approaches
                    depths = [surface_water_m[r, c] for r, c in cells]
                    effective_depth = float(max(depths)) if depths else 0.0
            else:
                # Standard surface road: sample max water depth along the segment
                depths = [surface_water_m[r, c] for r, c in cells]
                effective_depth = float(max(depths)) if depths else 0.0
                
            edge_data["current_water_depth_m"] = effective_depth
            
            # Passability flags
            edge_data["is_passable_passenger"] = bool(effective_depth < self.crit_depth_passenger)
            edge_data["is_passable_emergency"] = bool(effective_depth < self.crit_depth_emergency)
            edge_data["is_passable"] = edge_data["is_passable_passenger"]
            
            # Speed degradation model (Pregnolato et al. 2017)
            v0_mps = edge_data["free_flow_speed_mps"]
            if effective_depth < 0.02: # Dry / minor moisture
                v_flood_mps = v0_mps
            elif effective_depth >= self.crit_depth_emergency:
                v_flood_mps = 0.0 # Impassable
            else:
                ratio = (effective_depth - 0.02) / (self.crit_depth_emergency - 0.02)
                degradation = np.clip(1.0 - (ratio**1.4), 0.0, 1.0)
                v_flood_mps = v0_mps * degradation
                
            edge_data["flood_speed_mps"] = float(v_flood_mps)
            edge_data["current_speed_mps"] = float(v_flood_mps)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates global network operational health metrics."""
        total_edges = self.graph.number_of_edges()
        impassable_passenger = 0
        impassable_emergency = 0
        total_length_m = 0.0
        flooded_length_m = 0.0
        
        for _, _, data in self.graph.edges(data=True):
            length = data.get("length_m", 100.0)
            total_length_m += length
            if not data.get("is_passable_passenger", True):
                impassable_passenger += 1
                flooded_length_m += length
            if not data.get("is_passable_emergency", True):
                impassable_emergency += 1
                
        return {
            "total_edges": total_edges,
            "closed_edges_passenger": impassable_passenger,
            "closed_edges_emergency": impassable_emergency,
            "passenger_closure_pct": (impassable_passenger / max(1, total_edges)) * 100.0,
            "emergency_closure_pct": (impassable_emergency / max(1, total_edges)) * 100.0,
            "flooded_road_km": flooded_length_m / 1000.0,
            "total_network_km": total_length_m / 1000.0,
        }
