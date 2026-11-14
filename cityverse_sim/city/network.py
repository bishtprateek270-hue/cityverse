"""
cityverse_sim.city.network
==========================
Graph topology definitions for MetroVerdia:
- Surface Road Network (G_road): directed planar graph with road classes, bridges, and elevation.
- Subsurface Drainage Network (G_pipe): directed graph with manholes, conduits, pumps, and river outfalls.
- Critical Facilities Registry: hospitals, EMS dispatch stations, electrical substations.
"""

from typing import Dict, Any, List, Tuple
import networkx as nx
import numpy as np

from cityverse_sim.city.grid import CityGrid


class RoadNetwork:
    """Directed graph representing MetroVerdia's transport system."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_network(self, grid: CityGrid, road_config: Dict[str, Any]):
        """
        Constructs a realistic urban road hierarchy:
        - Primary East-West Highway (embanked)
        - North-South Arterial Corridors
        - Urban Core Grid (high density)
        - Bridges crossing the central river
        """
        self.graph.clear()
        
        # 1. Define intersection grid spacing
        # Urban Core (high density): 300m spacing
        # Outer Suburbs: 600m - 900m spacing
        xs_core = np.arange(2100.0, 4200.0 + 1.0, 300.0)
        ys_core = np.arange(1800.0, 3900.0 + 1.0, 300.0)
        
        xs_outer = np.arange(600.0, 5700.0 + 1.0, 600.0)
        ys_outer = np.arange(600.0, 5700.0 + 1.0, 600.0)
        
        # Combine unique node coordinates
        all_coords = set()
        for x in xs_core:
            for y in ys_core:
                all_coords.add((float(x), float(y)))
        for x in xs_outer:
            for y in ys_outer:
                all_coords.add((float(x), float(y)))
                
        sorted_coords = sorted(list(all_coords), key=lambda p: (p[0], p[1]))
        
        # Add Nodes
        node_id_map = {}
        for idx, (x, y) in enumerate(sorted_coords):
            node_id = f"INT-{idx:04d}"
            r, c = grid.coord_to_cell(x, y)
            elev = float(grid.dem[r, c])
            is_river = bool(grid.river_mask[r, c])
            
            self.graph.add_node(
                node_id,
                x=x,
                y=y,
                z=elev,
                grid_r=r,
                grid_c=c,
                is_river=is_river,
                node_type="intersection"
            )
            node_id_map[(round(x, 1), round(y, 1))] = node_id
            
        # Add Horizontal and Vertical Edges between adjacent coordinates
        # Group by y to find horizontal segments
        from collections import defaultdict
        by_y = defaultdict(list)
        by_x = defaultdict(list)
        for (x, y) in sorted_coords:
            by_y[round(y, 1)].append(x)
            by_x[round(x, 1)].append(y)
            
        edge_counter = 0
        
        # Horizontal roads
        for y_val, x_list in by_y.items():
            x_sorted = sorted(x_list)
            for i in range(len(x_sorted) - 1):
                x1, x2 = x_sorted[i], x_sorted[i+1]
                dist = x2 - x1
                if dist > 950.0:
                    continue # Gap too large for direct link
                u = node_id_map[(round(x1, 1), round(y_val, 1))]
                v = node_id_map[(round(x2, 1), round(y_val, 1))]
                
                # Check road classification
                # East-West Highway at y = 3000
                is_hwy = abs(y_val - 3000.0) < 50.0
                is_core = (2100.0 <= (x1+x2)/2 <= 4200.0) and (1800.0 <= y_val <= 3900.0)
                
                road_class = "highway" if is_hwy else ("arterial" if is_core else "local")
                is_bridge = self.graph.nodes[u]["is_river"] or self.graph.nodes[v]["is_river"]
                
                self._add_bidirectional_edge(
                    u, v, dist, road_class, is_bridge, road_config, edge_counter
                )
                edge_counter += 2

        # Vertical roads
        for x_val, y_list in by_x.items():
            y_sorted = sorted(y_list)
            for i in range(len(y_sorted) - 1):
                y1, y2 = y_sorted[i], y_sorted[i+1]
                dist = y2 - y1
                if dist > 950.0:
                    continue
                u = node_id_map[(round(x_val, 1), round(y1, 1))]
                v = node_id_map[(round(x_val, 1), round(y2, 1))]
                
                # North-South Arterial at x = 3000 or x = 4200
                is_art = abs(x_val - 3000.0) < 50.0 or abs(x_val - 4200.0) < 50.0
                is_core = (2100.0 <= x_val <= 4200.0) and (1800.0 <= (y1+y2)/2 <= 3900.0)
                road_class = "arterial" if (is_art or is_core) else "local"
                is_bridge = self.graph.nodes[u]["is_river"] or self.graph.nodes[v]["is_river"]
                
                self._add_bidirectional_edge(
                    u, v, dist, road_class, is_bridge, road_config, edge_counter
                )
                edge_counter += 2

    def _add_bidirectional_edge(
        self, u: str, v: str, length: float, road_class: str,
        is_bridge: bool, road_config: Dict[str, Any], counter: int
    ):
        cls_spec = road_config.get("road_classes", {}).get(
            road_class, {"lanes": 2, "free_flow_speed_kmh": 50.0, "capacity_veh_per_hr_lane": 1000}
        )
        ff_speed_kmh = float(cls_spec.get("free_flow_speed_kmh", 50.0))
        speed_mps = ff_speed_kmh / 3.6
        base_travel_time_s = length / max(speed_mps, 1.0)
        lanes = int(cls_spec.get("lanes", 2))
        cap_per_lane = float(cls_spec.get("capacity_veh_per_hr_lane", 1000))
        total_cap = lanes * cap_per_lane
        
        # Bridge elevation: elevated above ground level
        u_z = self.graph.nodes[u]["z"]
        v_z = self.graph.nodes[v]["z"]
        avg_z = (u_z + v_z) / 2.0
        if is_bridge:
            avg_z += 2.0 # 2m clearance above riverbed/banks
            
        # Direction 1: u -> v
        self.graph.add_edge(
            u, v,
            edge_id=f"ROAD-{counter:05d}",
            length_m=length,
            road_class=road_class,
            lanes=lanes,
            free_flow_speed_mps=speed_mps,
            free_flow_speed_kmh=ff_speed_kmh,
            capacity_veh_per_hr=total_cap,
            base_travel_time_s=base_travel_time_s,
            is_bridge=is_bridge,
            elevation_m=avg_z,
            current_water_depth_m=0.0,
            current_speed_mps=speed_mps,
            current_travel_time_s=base_travel_time_s,
            is_passable=True
        )
        # Direction 2: v -> u
        self.graph.add_edge(
            v, u,
            edge_id=f"ROAD-{counter+1:05d}",
            length_m=length,
            road_class=road_class,
            lanes=lanes,
            free_flow_speed_mps=speed_mps,
            free_flow_speed_kmh=ff_speed_kmh,
            capacity_veh_per_hr=total_cap,
            base_travel_time_s=base_travel_time_s,
            is_bridge=is_bridge,
            elevation_m=avg_z,
            current_water_depth_m=0.0,
            current_speed_mps=speed_mps,
            current_travel_time_s=base_travel_time_s,
            is_passable=True
        )

    def find_nearest_node(self, x: float, y: float) -> str:
        """Finds closest road intersection node to arbitrary (x, y) coordinates."""
        min_dist_sq = float("inf")
        closest_id = None
        for n, data in self.graph.nodes(data=True):
            dist_sq = (data["x"] - x)**2 + (data["y"] - y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_id = n
        return closest_id


class DrainageNetwork:
    """Directed graph representing subsurface stormwater conduits, junctions, and outfalls."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_network(self, grid: CityGrid, road_net: RoadNetwork):
        """
        Builds a stormwater drainage network aligned with roads and sloping toward river outfalls:
        - Inlets placed at urban intersections
        - Subsurface conduits conveying flow down hydraulic gradient
        - Discharge outfalls at the central river and south basin
        """
        self.graph.clear()
        
        # 1. Identify 3 River Outfall points along the river spline
        outfalls = [
            ("OUTFALL-1", 1800.0, 4400.0), # Upper reach outfall
            ("OUTFALL-2", 3400.0, 2700.0), # Central reach outfall
            ("OUTFALL-3", 5200.0, 1000.0), # Lowland alluvial outfall
        ]
        
        for oid, ox, oy in outfalls:
            r, c = grid.coord_to_cell(ox, oy)
            z_surf = float(grid.dem[r, c])
            self.graph.add_node(
                oid,
                x=ox,
                y=oy,
                z_ground=z_surf,
                invert_elevation=z_surf - 1.5,
                node_type="outfall",
                head_m=0.0,
                surcharge_m=0.0
            )
            
        # 2. Convert urban road intersections into catch basins/manholes
        pipe_counter = 0
        for n, data in road_net.graph.nodes(data=True):
            # Only place stormwater catch basins in urban core or lower elevations
            if data["z"] > 52.0:
                continue # Highland greenbelt drains naturally via overland swales
                
            manhole_id = f"MH-{n}"
            z_surf = data["z"]
            self.graph.add_node(
                manhole_id,
                x=data["x"],
                y=data["y"],
                z_ground=z_surf,
                invert_elevation=z_surf - 2.5, # 2.5m bury depth
                node_type="junction",
                head_m=0.0,
                surcharge_m=0.0,
                inlet_curb_length_m=1.2
            )

        # 3. Create conduits connecting manholes downward toward outfalls
        nodes = list(self.graph.nodes(data=True))
        for u, u_data in nodes:
            if u_data["node_type"] == "outfall":
                continue
                
            # Find closest candidate downhill node (another manhole or outfall)
            ux, uy, uz = u_data["x"], u_data["y"], u_data["invert_elevation"]
            best_target = None
            min_dist = float("inf")
            
            for v, v_data in nodes:
                if u == v:
                    continue
                vx, vy, vz = v_data["x"], v_data["y"], v_data["invert_elevation"]
                dist = np.hypot(vx - ux, vy - uy)
                
                # Must be reasonably close (< 650m) and downhill
                if dist < 650.0 and vz < uz:
                    if dist < min_dist:
                        min_dist = dist
                        best_target = v
                        
            if best_target:
                v_data = self.graph.nodes[best_target]
                slope = max(0.001, (u_data["invert_elevation"] - v_data["invert_elevation"]) / min_dist)
                
                # Diameter: 0.6m for standard, 1.2m for trunk lines near outfalls
                is_trunk = (v_data["node_type"] == "outfall") or (min_dist > 400.0)
                diameter_m = 1.4 if is_trunk else 0.75
                area = np.pi * (diameter_m / 2.0)**2
                hyd_radius = diameter_m / 4.0
                manning_n = 0.013 # Smooth concrete pipe
                
                # Manning capacity full pipe: Q = (1/n) * A * R^(2/3) * S^(1/2)
                full_capacity_m3_s = (1.0 / manning_n) * area * (hyd_radius**(2.0/3.0)) * np.sqrt(slope)
                
                self.graph.add_edge(
                    u, best_target,
                    edge_id=f"PIPE-{pipe_counter:04d}",
                    length_m=min_dist,
                    diameter_m=diameter_m,
                    slope=slope,
                    manning_n=manning_n,
                    capacity_m3_s=full_capacity_m3_s,
                    current_flow_m3_s=0.0,
                    capacity_utilization=0.0,
                    is_blocked=False
                )
                pipe_counter += 1


class FacilityRegistry:
    """Manages critical infrastructure facilities and their spatial connections."""

    def __init__(self):
        self.facilities = {}

    def register_from_config(self, config_infra: Dict[str, Any], road_net: RoadNetwork):
        """Associates all hospitals, EMS bases, and substations with the nearest road node."""
        self.facilities.clear()
        
        for category in ["hospitals", "ems_stations", "electrical_substations", "pumping_stations"]:
            items = config_infra.get(category, [])
            for item in items:
                fid = item["id"]
                x, y = float(item["x"]), float(item["y"])
                nearest_road_node = road_net.find_nearest_node(x, y)
                road_node_data = road_net.graph.nodes[nearest_road_node]
                
                facility_record = dict(item)
                facility_record["category"] = category
                facility_record["nearest_road_node"] = nearest_road_node
                facility_record["elevation_m"] = road_node_data["z"]
                facility_record["is_operational"] = True
                facility_record["flood_depth_m"] = 0.0
                
                self.facilities[fid] = facility_record

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [f for f in self.facilities.values() if f.get("category") == category]

