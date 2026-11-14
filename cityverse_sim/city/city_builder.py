"""
cityverse_sim.city.city_builder
===============================
Master procedural builder for MetroVerdia.
Loads city_spec.yaml, instantiates CityGrid, RoadNetwork, DrainageNetwork, FacilityRegistry,
validates hydraulic and topological constraints, and exports GIS-standard files.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from cityverse_sim.city.grid import CityGrid
from cityverse_sim.city.network import RoadNetwork, DrainageNetwork, FacilityRegistry


class MetroVerdiaCity:
    """Complete container representing the synthetic city environment."""

    def __init__(self, config_path: str = "configs/city_spec.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        spatial_cfg = self.config.get("spatial", {})
        self.grid = CityGrid(
            extent_x_m=float(spatial_cfg.get("extent_x_m", 6000.0)),
            extent_y_m=float(spatial_cfg.get("extent_y_m", 6000.0)),
            resolution_m=float(spatial_cfg.get("grid_resolution_m", 50.0)),
        )
        self.roads = RoadNetwork()
        self.drainage = DrainageNetwork()
        self.facilities = FacilityRegistry()

    def build(self, seed: int = 42):
        """Constructs all layers in topological sequence."""
        # 1. Generate Terrain & River Incision
        topo_cfg = self.config.get("topography", {})
        self.grid.generate_terrain(
            min_elev=float(topo_cfg.get("elevation_min_m", 12.0)),
            max_elev=float(topo_cfg.get("elevation_max_m", 68.0)),
            seed=seed
        )
        
        # 2. Assign Land Use & Infiltration Properties
        self.grid.assign_zones(self.config.get("zones", {}))
        
        # 3. Build Road Network
        self.roads.build_network(self.grid, self.config.get("transportation", {}))
        
        # 4. Build Drainage Subsurface Network
        self.drainage.build_network(self.grid, self.roads)
        
        # 5. Register Critical Facilities
        self.facilities.register_from_config(self.config.get("infrastructure", {}), self.roads)

    def export_gis_data(self, output_dir: str = "data/raw_city_gis"):
        """Exports layers to GeoJSON and NumPy binary arrays."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Export DEM and Grid Arrays
        np.save(out_path / "dem_elevation.npy", self.grid.dem)
        np.save(out_path / "manning_n.npy", self.grid.manning_n)
        np.save(out_path / "impervious_fraction.npy", self.grid.impervious_fraction)
        np.save(out_path / "river_mask.npy", self.grid.river_mask)
        
        # Grid metadata
        grid_meta = {
            "extent_x_m": self.grid.extent_x,
            "extent_y_m": self.grid.extent_y,
            "dx": self.grid.dx,
            "dy": self.grid.dy,
            "rows": self.grid.rows,
            "cols": self.grid.cols,
            "min_elevation_m": float(self.grid.dem.min()),
            "max_elevation_m": float(self.grid.dem.max()),
            "mean_elevation_m": float(self.grid.dem.mean()),
            "river_cell_count": int(self.grid.river_mask.sum())
        }
        with open(out_path / "dem_metadata.json", "w", encoding="utf-8") as f:
            json.dump(grid_meta, f, indent=2)
            
        # 2. Export Road Network GeoJSON
        road_features = []
        for u, v, data in self.roads.graph.edges(data=True):
            u_node = self.roads.graph.nodes[u]
            v_node = self.roads.graph.nodes[v]
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [u_node["x"], u_node["y"]],
                        [v_node["x"], v_node["y"]]
                    ]
                },
                "properties": {
                    "edge_id": data["edge_id"],
                    "u": u,
                    "v": v,
                    "road_class": data["road_class"],
                    "lanes": data["lanes"],
                    "length_m": data["length_m"],
                    "free_flow_speed_kmh": data["free_flow_speed_kmh"],
                    "capacity_veh_per_hr": data["capacity_veh_per_hr"],
                    "is_bridge": data["is_bridge"],
                    "elevation_m": data["elevation_m"]
                }
            }
            road_features.append(feature)
            
        roads_geojson = {"type": "FeatureCollection", "features": road_features}
        with open(out_path / "road_network.geojson", "w", encoding="utf-8") as f:
            json.dump(roads_geojson, f, indent=2)
            
        # 3. Export Drainage Conduits GeoJSON
        pipe_features = []
        for u, v, data in self.drainage.graph.edges(data=True):
            u_node = self.drainage.graph.nodes[u]
            v_node = self.drainage.graph.nodes[v]
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [u_node["x"], u_node["y"]],
                        [v_node["x"], v_node["y"]]
                    ]
                },
                "properties": {
                    "edge_id": data["edge_id"],
                    "from_node": u,
                    "to_node": v,
                    "length_m": data["length_m"],
                    "diameter_m": data["diameter_m"],
                    "slope": data["slope"],
                    "capacity_m3_s": data["capacity_m3_s"]
                }
            }
            pipe_features.append(feature)
            
        drainage_geojson = {"type": "FeatureCollection", "features": pipe_features}
        with open(out_path / "drainage_conduits.geojson", "w", encoding="utf-8") as f:
            json.dump(drainage_geojson, f, indent=2)
            
        # 4. Export Critical Facilities GeoJSON
        facility_features = []
        for fid, fac in self.facilities.facilities.items():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [fac["x"], fac["y"]]
                },
                "properties": fac
            }
            facility_features.append(feature)
            
        facilities_geojson = {"type": "FeatureCollection", "features": facility_features}
        with open(out_path / "critical_facilities.geojson", "w", encoding="utf-8") as f:
            json.dump(facilities_geojson, f, indent=2)
            
        # 5. Export Master Summary
        summary = {
            "city_name": self.config.get("city_name", "MetroVerdia"),
            "grid_dimensions": [self.grid.rows, self.grid.cols],
            "total_road_nodes": self.roads.graph.number_of_nodes(),
            "total_road_directed_edges": self.roads.graph.number_of_edges(),
            "bridge_count": sum(1 for _, _, d in self.roads.graph.edges(data=True) if d.get("is_bridge")),
            "drainage_manholes": sum(1 for _, d in self.drainage.graph.nodes(data=True) if d.get("node_type") == "junction"),
            "drainage_outfalls": sum(1 for _, d in self.drainage.graph.nodes(data=True) if d.get("node_type") == "outfall"),
            "drainage_conduits": self.drainage.graph.number_of_edges(),
            "registered_facilities": len(self.facilities.facilities),
            "hospitals": len(self.facilities.get_by_category("hospitals")),
            "ems_stations": len(self.facilities.get_by_category("ems_stations")),
            "substations": len(self.facilities.get_by_category("electrical_substations")),
            "pumping_stations": len(self.facilities.get_by_category("pumping_stations"))
        }
        with open(out_path / "city_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        return summary


def build_and_export_metroverdia(config_path: str = "configs/city_spec.yaml", output_dir: str = "data/raw_city_gis"):
    """Command-line / script entry point."""
    city = MetroVerdiaCity(config_path=config_path)
    city.build()
    summary = city.export_gis_data(output_dir=output_dir)
    print("MetroVerdia procedural construction complete:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")
    return city, summary


if __name__ == "__main__":
    build_and_export_metroverdia()

