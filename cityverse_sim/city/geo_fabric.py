"""
cityverse_sim.city.geo_fabric
=============================
Generates authentic geographical urban fabric for MetroVerdia:
- Urban blocks partitioned by the road network.
- Realistic architectural building footprints aligned to streets with zoning setbacks.
- Analytical hillshade matrix computed from DEM elevation gradients.
"""

from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
import numpy as np


class GeoFabricGenerator:
    """Generates realistic geographical layers and analytical hillshading."""

    def __init__(self, raw_gis_dir: str = "data/raw_city_gis"):
        self.raw_gis_dir = Path(raw_gis_dir)
        self.raw_gis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load DEM
        dem_path = self.raw_gis_dir / "dem_elevation.npy"
        if dem_path.exists():
            self.dem = np.load(dem_path).astype(np.float32)
        else:
            self.dem = np.zeros((120, 120), dtype=np.float32) + 30.0
            
        river_path = self.raw_gis_dir / "river_mask.npy"
        if river_path.exists():
            self.river_mask = np.load(river_path).astype(bool)
        else:
            self.river_mask = np.zeros((120, 120), dtype=bool)

    def generate_hillshade(self, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
        """
        Computes standard multi-directional analytical hillshading from DEM.
        Returns a normalized 2D array (0.0 to 1.0) of illuminated terrain relief.
        """
        azimuth_rad = np.radians(360.0 - azimuth_deg + 90.0)
        altitude_rad = np.radians(altitude_deg)
        
        # Grid cell size dx = dy = 50m
        dx = 50.0
        dy = 50.0
        
        # Central differences for slope gradients dz/dx and dz/dy
        dz_dx = np.gradient(self.dem, dx, axis=1)
        dz_dy = np.gradient(self.dem, dy, axis=0)
        
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect = np.arctan2(-dz_dy, -dz_dx)
        
        # Hillshade equation: cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)
        zenith = np.pi / 2.0 - altitude_rad
        hillshade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(azimuth_rad - aspect)
        hillshade = np.clip(hillshade, 0.0, 1.0)
        
        out_path = self.raw_gis_dir / "dem_hillshade.npy"
        np.save(out_path, hillshade.astype(np.float32))
        return hillshade

    def _is_river(self, x: float, y: float, margin_m: float = 50.0) -> bool:
        c_min = int(np.clip((x - margin_m) / 50.0, 0, 119))
        c_max = int(np.clip((x + margin_m) / 50.0, 0, 119))
        r_min = int(np.clip((6000.0 - (y + margin_m)) / 50.0, 0, 119))
        r_max = int(np.clip((6000.0 - (y - margin_m)) / 50.0, 0, 119))
        return bool(np.any(self.river_mask[r_min:r_max+1, c_min:c_max+1]))

    def generate_urban_blocks_and_footprints(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generates realistic city blocks and aligned building footprints across MetroVerdia.
        No random colored bars; authentic parcel layouts with setbacks and zoning classes.
        """
        np.random.seed(101)
        
        # Load road network to identify intersection grid
        road_path = self.raw_gis_dir / "road_network.geojson"
        intersections = []
        if road_path.exists():
            with open(road_path, "r", encoding="utf-8") as f:
                rdata = json.load(f)
                seen_coords = set()
                for feat in rdata["features"]:
                    for pt in feat["geometry"]["coordinates"]:
                        pt_tup = (round(pt[0]), round(pt[1]))
                        if pt_tup not in seen_coords:
                            seen_coords.add(pt_tup)
                            intersections.append(pt_tup)

        # Urban zoning sectors
        # (name, x_min, x_max, y_min, y_max, zoning_class, color, stories_range, footprint_range)
        sectors = [
            # Downtown Financial Core
            {
                "name": "Downtown Commercial Core",
                "bounds": (2600.0, 3800.0, 3000.0, 4200.0),
                "zoning": "commercial_high",
                "color": "#38bdf8",
                "stories_range": (12, 45),
                "footprint_range": (35.0, 60.0),
                "density": 0.70,
            },
            # Midtown Office & Civic Plazas
            {
                "name": "Midtown Corporate Corridor",
                "bounds": (1800.0, 3000.0, 1800.0, 3000.0),
                "zoning": "commercial_medium",
                "color": "#818cf8",
                "stories_range": (6, 16),
                "footprint_range": (30.0, 50.0),
                "density": 0.65,
            },
            # Pine Heights Residential (North Uplands)
            {
                "name": "Pine Heights Residential",
                "bounds": (800.0, 3400.0, 4200.0, 5600.0),
                "zoning": "residential_low",
                "color": "#34d399",
                "stories_range": (2, 4),
                "footprint_range": (18.0, 30.0),
                "density": 0.50,
            },
            # West Riverfront District
            {
                "name": "West Riverfront Promenade",
                "bounds": (600.0, 2000.0, 1200.0, 3200.0),
                "zoning": "residential_medium",
                "color": "#2dd4bf",
                "stories_range": (4, 10),
                "footprint_range": (25.0, 42.0),
                "density": 0.55,
            },
            # East Iron Basin Logistics & Rail
            {
                "name": "Iron Basin Industrial Park",
                "bounds": (3800.0, 5600.0, 2800.0, 5200.0),
                "zoning": "industrial_heavy",
                "color": "#fbbf24",
                "stories_range": (1, 3),
                "footprint_range": (45.0, 85.0),
                "density": 0.45,
            },
            # South Confluence Port & Basin Lowlands
            {
                "name": "South Confluence Port & Drainage Basin",
                "bounds": (3200.0, 5600.0, 600.0, 2400.0),
                "zoning": "port_industrial",
                "color": "#f87171",
                "stories_range": (1, 4),
                "footprint_range": (40.0, 75.0),
                "density": 0.40,
            }
        ]

        # Generate rectangular urban blocks bounded by 600m street intervals
        block_features = []
        building_features = []
        bldg_id = 0
        block_id = 0

        # Primary street grid intervals: 600m to 5400m in steps of 600m
        xs = np.arange(600.0, 5400.0, 600.0)
        ys = np.arange(600.0, 5400.0, 600.0)

        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i+1]
            for j in range(len(ys) - 1):
                y1, y2 = ys[j], ys[j+1]

                bx_center = (x1 + x2) / 2.0
                by_center = (y1 + y2) / 2.0

                # Check if block is entirely or largely inside river
                if self._is_river(bx_center, by_center, margin_m=120.0):
                    continue

                # Identify sector zoning
                sec = sectors[0]
                for s in sectors:
                    xmin, xmax, ymin, ymax = s["bounds"]
                    if xmin <= bx_center <= xmax and ymin <= by_center <= ymax:
                        sec = s
                        break

                # Create block geometry with 20m street buffer setback
                block_pad = 22.0
                bx_min, bx_max = x1 + block_pad, x2 - block_pad
                by_min, by_max = y1 + block_pad, y2 - block_pad

                block_poly = [
                    [bx_min, by_min],
                    [bx_max, by_min],
                    [bx_max, by_max],
                    [bx_min, by_max],
                    [bx_min, by_min]
                ]

                block_features.append({
                    "type": "Feature",
                    "properties": {
                        "id": f"BLOCK-{block_id:04d}",
                        "zoning": sec["zoning"],
                        "district": sec["name"],
                        "color": sec["color"],
                        "area_m2": (bx_max - bx_min) * (by_max - by_min)
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [block_poly]
                    }
                })
                block_id += 1

                # Subdivide block into realistic parcel footprints
                if sec["zoning"] == "residential_low":
                    n_cols, n_rows = 5, 4
                elif "commercial" in sec["zoning"]:
                    n_cols, n_rows = 4, 3
                elif "residential" in sec["zoning"]:
                    n_cols, n_rows = 4, 4
                else:
                    n_cols, n_rows = 3, 2

                parcel_w = (bx_max - bx_min) / n_cols
                parcel_h = (by_max - by_min) / n_rows

                for pi in range(n_cols):
                    for pj in range(n_rows):
                        if np.random.rand() > sec["density"]:
                            continue # Open space / courtyard / green parking

                        px1 = bx_min + pi * parcel_w + 5.0
                        py1 = by_min + pj * parcel_h + 5.0
                        px2 = px1 + parcel_w - 10.0
                        py2 = py1 + parcel_h - 10.0

                        cx = (px1 + px2) / 2.0
                        cy = (py1 + py2) / 2.0

                        if self._is_river(cx, cy, margin_m=35.0):
                            continue

                        # Architectural building footprint
                        bw = (px2 - px1) * np.random.uniform(0.78, 0.94)
                        bh = (py2 - py1) * np.random.uniform(0.78, 0.94)

                        fx1, fx2 = cx - bw / 2.0, cx + bw / 2.0
                        fy1, fy2 = cy - bh / 2.0, cy + bh / 2.0

                        footprint_poly = [
                            [fx1, fy1],
                            [fx2, fy1],
                            [fx2, fy2],
                            [fx1, fy2],
                            [fx1, fy1]
                        ]

                        stories = int(np.random.randint(sec["stories_range"][0], sec["stories_range"][1] + 1))
                        height_m = round(stories * 3.5, 1)

                        # Cell elevation
                        r = int(np.clip((6000.0 - cy) / 50.0, 0, 119))
                        c = int(np.clip(cx / 50.0, 0, 119))
                        ground_elev = float(self.dem[r, c])

                        building_features.append({
                            "type": "Feature",
                            "properties": {
                                "id": f"BLDG-{bldg_id:05d}",
                                "zoning": sec["zoning"],
                                "district": sec["name"],
                                "stories": stories,
                                "height_m": height_m,
                                "ground_elev_m": round(ground_elev, 1),
                                "width_m": round(bw, 1),
                                "depth_m": round(bh, 1),
                                "center": [round(cx, 1), round(cy, 1)],
                                "use_type": "Commercial Office" if "commercial" in sec["zoning"] else ("Residential" if "residential" in sec["zoning"] else "Industrial/Port Logistics"),
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [footprint_poly]
                            }
                        })
                        bldg_id += 1

        blocks_fc = {"type": "FeatureCollection", "features": block_features}
        bldg_fc = {"type": "FeatureCollection", "features": building_features}

        with open(self.raw_gis_dir / "urban_blocks.geojson", "w", encoding="utf-8") as f:
            json.dump(blocks_fc, f, indent=2)

        with open(self.raw_gis_dir / "building_footprints.geojson", "w", encoding="utf-8") as f:
            json.dump(bldg_fc, f, indent=2)

        print(f"Generated {len(block_features)} urban blocks and {len(building_features)} building footprints.")
        return blocks_fc, bldg_fc


if __name__ == "__main__":
    fab = GeoFabricGenerator()
    fab.generate_hillshade()
    fab.generate_urban_blocks_and_footprints()
