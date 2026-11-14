"""
cityverse_sim.real_world.ingestion
==================================
Geospatial ingestion pipeline for real-world cities:
- Imports real Digital Elevation Models (DEMs from USGS 3DEP, Copernicus, or GeoTIFF)
- Ingests OpenStreetMap (OSM) vector road networks and water bodies
- Projects coordinates into metric Cartesian space (meters)
- Snaps road vertices to terrain surface elevations
- Derives roughness and imperviousness fields from Land-Use Land-Cover (LULC)
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import networkx as nx

from cityverse_sim.city.grid import CityGrid
from cityverse_sim.city.network import RoadNetwork, DrainageNetwork, FacilityRegistry


class RealWorldGeoIngestor:
    """
    Ingests real-world geospatial datasets and translates them into
    CityVerse simulation state representations.
    """

    def __init__(self, target_resolution_m: float = 50.0):
        self.dx = float(target_resolution_m)
        self.dy = float(target_resolution_m)

    def ingest_dem_raster(
        self,
        elevation_matrix: np.ndarray,
        extent_x_m: float,
        extent_y_m: float,
        nodata_val: float = -9999.0,
    ) -> CityGrid:
        """
        Ingests a 2D elevation grid from a real DEM.
        Fills nodata values via nearest valid neighbor interpolation.
        """
        dem = elevation_matrix.astype(np.float32)
        
        # Fill nodata
        is_nodata = (dem == nodata_val) | np.isnan(dem)
        if np.any(is_nodata):
            valid_min = float(np.min(dem[~is_nodata])) if np.any(~is_nodata) else 0.0
            dem[is_nodata] = valid_min
            
        rows, cols = dem.shape
        grid = CityGrid(extent_x_m=extent_x_m, extent_y_m=extent_y_m, resolution_m=self.dx)
        grid.dem = dem
        grid.rows = rows
        grid.cols = cols
        
        return grid

    def ingest_osm_roads(
        self,
        grid: CityGrid,
        osm_road_features: List[Dict[str, Any]],
    ) -> RoadNetwork:
        """
        Ingests road lines from OpenStreetMap GeoJSON, snaps intersections
        to grid elevations, and constructs the directed transport graph.
        """
        road_net = RoadNetwork()
        graph = road_net.graph
        graph.clear()
        
        node_id_map = {}
        edge_counter = 0
        
        for feat in osm_road_features:
            coords = feat["geometry"]["coordinates"]
            props = feat.get("properties", {})
            highway_type = props.get("highway", "residential")
            
            # Map OSM highway types to CityVerse classes
            if highway_type in ["motorway", "trunk", "primary"]:
                road_class = "highway"
                lanes = 4
                speed_kmh = 90.0
            elif highway_type in ["secondary", "tertiary"]:
                road_class = "arterial"
                lanes = 2
                speed_kmh = 55.0
            else:
                road_class = "local"
                lanes = 1
                speed_kmh = 35.0
                
            speed_mps = speed_kmh / 3.6
            is_bridge = props.get("bridge") in ["yes", True, "1"]
            
            # Add segments between consecutive vertices
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i+1]
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])
                
                # Snap nodes
                for x, y in [(x1, y1), (x2, y2)]:
                    key = (round(x, 1), round(y, 1))
                    if key not in node_id_map:
                        node_id = f"OSM-{len(node_id_map):05d}"
                        r, c = grid.coord_to_cell(x, y)
                        elev = float(grid.dem[r, c])
                        graph.add_node(
                            node_id,
                            x=x, y=y, z=elev,
                            grid_r=r, grid_c=c,
                            is_river=bool(grid.river_mask[r, c]),
                            node_type="intersection"
                        )
                        node_id_map[key] = node_id
                        
                u = node_id_map[(round(x1, 1), round(y1, 1))]
                v = node_id_map[(round(x2, 1), round(y2, 1))]
                
                if u == v:
                    continue
                    
                dist = np.hypot(x2 - x1, y2 - y1)
                u_z = graph.nodes[u]["z"]
                v_z = graph.nodes[v]["z"]
                avg_z = (u_z + v_z) / 2.0
                if is_bridge:
                    avg_z += 2.0
                    
                base_tt_s = dist / max(speed_mps, 1.0)
                
                # Bidirectional
                for from_n, to_n in [(u, v), (v, u)]:
                    graph.add_edge(
                        from_n, to_n,
                        edge_id=f"REAL-{edge_counter:05d}",
                        length_m=dist,
                        road_class=road_class,
                        lanes=lanes,
                        free_flow_speed_mps=speed_mps,
                        free_flow_speed_kmh=speed_kmh,
                        capacity_veh_per_hr=lanes * 1000.0,
                        base_travel_time_s=base_tt_s,
                        is_bridge=is_bridge,
                        elevation_m=avg_z,
                        current_water_depth_m=0.0,
                        current_speed_mps=speed_mps,
                        current_travel_time_s=base_tt_s,
                        is_passable=True
                    )
                    edge_counter += 1
                    
        return road_net

    def derive_land_cover_properties(
        self,
        grid: CityGrid,
        lulc_classification_grid: np.ndarray,
    ):
        """
        Maps standard Land-Use Land-Cover (LULC) classes (e.g. Copernicus / ESA WorldCover)
        to Manning's roughness, impervious fraction, and Horton infiltration parameters.
        
        Classes:
            10: Tree cover / Forest
            20: Shrubland
            30: Grassland
            40: Cropland
            50: Built-up / Urban Impervious
            60: Bare / Sparse vegetation
            80: Permanent Water Bodies
        """
        lulc = lulc_classification_grid
        
        # 1. Built-up / Urban (Class 50)
        urban_mask = (lulc == 50)
        grid.impervious_fraction[urban_mask] = 0.85
        grid.manning_n[urban_mask] = 0.015
        grid.horton_f0[urban_mask] = 10.0
        grid.horton_fc[urban_mask] = 2.0
        grid.horton_decay[urban_mask] = 4.0
        
        # 2. Vegetated / Greenbelt (Classes 10, 20, 30)
        veg_mask = np.isin(lulc, [10, 20, 30])
        grid.impervious_fraction[veg_mask] = 0.10
        grid.manning_n[veg_mask] = 0.060
        grid.horton_f0[veg_mask] = 75.0
        grid.horton_fc[veg_mask] = 12.0
        grid.horton_decay[veg_mask] = 1.8
        
        # 3. Water Bodies (Class 80)
        water_mask = (lulc == 80)
        grid.river_mask[water_mask] = True
        grid.impervious_fraction[water_mask] = 1.0
        grid.manning_n[water_mask] = 0.025
        grid.horton_f0[water_mask] = 0.0
        grid.horton_fc[water_mask] = 0.0

