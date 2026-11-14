"""
cityverse_sim.city.urban_generator
===================================
Generates high-fidelity urban morphology for MetroVerdia:
- Over 1,000 architectural building footprints with heights, stories, zoning classes, and roof features.
- City district zoning boundaries (Downtown Core, Midtown, West Riverfront, Pine Heights, Industrial Basin, South Confluence).
- Green parks, waterfront esplanades, and landmark campus facilities.
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import numpy as np


class UrbanMorphologyGenerator:
    """Procedurally synthesizes realistic urban architectural features for MetroVerdia."""

    def __init__(self, raw_gis_dir: str = "data/raw_city_gis"):
        self.raw_gis_dir = Path(raw_gis_dir)
        self.raw_gis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load DEM if present to stamp building base elevations
        dem_path = self.raw_gis_dir / "dem_elevation.npy"
        if dem_path.exists():
            self.dem = np.load(dem_path)
        else:
            self.dem = np.zeros((120, 120), dtype=np.float32) + 30.0
            
        river_path = self.raw_gis_dir / "river_mask.npy"
        if river_path.exists():
            self.river_mask = np.load(river_path)
        else:
            self.river_mask = np.zeros((120, 120), dtype=bool)

    def _get_elev(self, x: float, y: float) -> float:
        c = int(np.clip(x / 50.0, 0, 119))
        r = int(np.clip((6000.0 - y) / 50.0, 0, 119))
        return float(self.dem[r, c])

    def _is_river(self, x: float, y: float, margin_m: float = 60.0) -> bool:
        c_min = int(np.clip((x - margin_m) / 50.0, 0, 119))
        c_max = int(np.clip((x + margin_m) / 50.0, 0, 119))
        r_min = int(np.clip((6000.0 - (y + margin_m)) / 50.0, 0, 119))
        r_max = int(np.clip((6000.0 - (y - margin_m)) / 50.0, 0, 119))
        return bool(np.any(self.river_mask[r_min:r_max+1, c_min:c_max+1]))

    def generate_districts(self) -> Dict[str, Any]:
        """Defines the 6 canonical urban districts of MetroVerdia."""
        districts = [
            {
                "id": "DIST-DOWNTOWN",
                "name": "Downtown Verdia",
                "type": "commercial_high_density",
                "center": [3200.0, 3600.0],
                "color": "#38bdf8",
                "description": "High-rise commercial, banking, and governmental towers.",
                "bounds": [2400.0, 2800.0, 4200.0, 4400.0] # [xmin, ymin, xmax, ymax]
            },
            {
                "id": "DIST-MIDTOWN",
                "name": "Midtown & Innovation Corridor",
                "type": "commercial_mixed",
                "center": [2400.0, 2600.0],
                "color": "#818cf8",
                "description": "Mid-rise corporate plazas, university facilities, and tech hubs.",
                "bounds": [1800.0, 1800.0, 3000.0, 3200.0]
            },
            {
                "id": "DIST-RIVERFRONT",
                "name": "West Riverfront Promenade",
                "type": "residential_waterfront",
                "center": [1400.0, 2200.0],
                "color": "#2dd4bf",
                "description": "Waterfront residences, parks, marinas, and river boardwalks.",
                "bounds": [600.0, 1400.0, 2000.0, 3000.0]
            },
            {
                "id": "DIST-PINE-HEIGHTS",
                "name": "Pine Heights Residential",
                "type": "residential_low_density",
                "center": [2200.0, 4800.0],
                "color": "#34d399",
                "description": "Elevated residential neighborhoods, schools, and civic green spaces.",
                "bounds": [800.0, 4000.0, 3400.0, 5600.0]
            },
            {
                "id": "DIST-EAST-INDUSTRIAL",
                "name": "Iron Basin Logistics Park",
                "type": "industrial_heavy",
                "center": [4600.0, 3800.0],
                "color": "#fbbf24",
                "description": "Manufacturing plants, distribution warehouses, and rail yards.",
                "bounds": [3800.0, 2800.0, 5600.0, 5200.0]
            },
            {
                "id": "DIST-SOUTH-CONFLUENCE",
                "name": "South Confluence & Harbor Basin",
                "type": "port_industrial_lowland",
                "center": [4400.0, 1400.0],
                "color": "#f87171",
                "description": "Flood vulnerable port basins, pump stations, and drainage canals.",
                "bounds": [3200.0, 600.0, 5600.0, 2400.0]
            }
        ]
        
        features = []
        for d in districts:
            xmin, ymin, xmax, ymax = d["bounds"]
            poly = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin]
            ]
            feat = {
                "type": "Feature",
                "properties": {
                    "id": d["id"],
                    "name": d["name"],
                    "district_type": d["type"],
                    "center": d["center"],
                    "color": d["color"],
                    "description": d["description"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [poly]
                }
            }
            features.append(feat)
            
        fc = {"type": "FeatureCollection", "features": features}
        out_path = self.raw_gis_dir / "urban_districts.geojson"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)
            
        return fc

    def generate_buildings(self) -> Dict[str, Any]:
        """Synthesizes over 1,200 realistic building footprints with architectural properties."""
        np.random.seed(42)
        features = []
        b_idx = 0

        # District zoning parameters
        # (center_x, center_y, radius_x, radius_y, count, type, height_min, height_max, width_range, color_palette)
        district_specs = [
            # Downtown Skyscraper Core
            {
                "name": "Downtown Core",
                "district_id": "DIST-DOWNTOWN",
                "cx": 3300.0, "cy": 3700.0, "rx": 700.0, "ry": 600.0,
                "count": 160,
                "types": ["skyscraper", "highrise_office", "luxury_tower"],
                "h_min": 50.0, "h_max": 160.0, "w_range": (35.0, 65.0),
                "colors": ["#0284c7", "#0369a1", "#075985", "#38bdf8", "#0ea5e9"]
            },
            # Midtown & Corporate Plazas
            {
                "name": "Midtown",
                "district_id": "DIST-MIDTOWN",
                "cx": 2500.0, "cy": 2500.0, "rx": 600.0, "ry": 600.0,
                "count": 180,
                "types": ["midrise_commercial", "corporate_office", "retail_plaza"],
                "h_min": 24.0, "h_max": 65.0, "w_range": (30.0, 50.0),
                "colors": ["#4f46e5", "#4338ca", "#6366f1", "#818cf8"]
            },
            # West Riverfront Residential & Mixed
            {
                "name": "West Riverfront",
                "district_id": "DIST-RIVERFRONT",
                "cx": 1300.0, "cy": 2300.0, "rx": 550.0, "ry": 700.0,
                "count": 190,
                "types": ["waterfront_condo", "apartment_block", "marina_complex"],
                "h_min": 15.0, "h_max": 45.0, "w_range": (25.0, 45.0),
                "colors": ["#0d9488", "#0f766e", "#14b8a6", "#2dd4bf"]
            },
            # Pine Heights Residential (Homes & Low-rise)
            {
                "name": "Pine Heights",
                "district_id": "DIST-PINE-HEIGHTS",
                "cx": 2100.0, "cy": 4800.0, "rx": 900.0, "ry": 700.0,
                "count": 320,
                "types": ["suburban_residence", "duplex_home", "neighborhood_school", "civic_building"],
                "h_min": 8.0, "h_max": 20.0, "w_range": (18.0, 32.0),
                "colors": ["#059669", "#047857", "#10b981", "#34d399", "#d97706"]
            },
            # East Industrial & Logistics
            {
                "name": "Iron Basin Logistics",
                "district_id": "DIST-EAST-INDUSTRIAL",
                "cx": 4700.0, "cy": 3900.0, "rx": 800.0, "ry": 800.0,
                "count": 180,
                "types": ["warehouse", "factory_plant", "logistics_depot", "fuel_silo"],
                "h_min": 12.0, "h_max": 30.0, "w_range": (45.0, 85.0),
                "colors": ["#d97706", "#b45309", "#92400e", "#78350f", "#64748b"]
            },
            # South Basin Port & Drainage Works
            {
                "name": "South Confluence",
                "district_id": "DIST-SOUTH-CONFLUENCE",
                "cx": 4500.0, "cy": 1500.0, "rx": 700.0, "ry": 600.0,
                "count": 170,
                "types": ["port_shed", "freight_terminal", "cargo_facility", "utility_plant"],
                "h_min": 10.0, "h_max": 28.0, "w_range": (35.0, 75.0),
                "colors": ["#e11d48", "#be123c", "#9f1239", "#475569", "#334155"]
            }
        ]

        for spec in district_specs:
            placed = 0
            attempts = 0
            max_attempts = spec["count"] * 5
            
            while placed < spec["count"] and attempts < max_attempts:
                attempts += 1
                # Gaussian jitter around district center
                bx = float(np.random.normal(spec["cx"], spec["rx"] * 0.45))
                by = float(np.random.normal(spec["cy"], spec["ry"] * 0.45))
                
                # Keep within bounds of 6km x 6km
                if bx < 300.0 or bx > 5700.0 or by < 300.0 or by > 5700.0:
                    continue
                    
                # Do not place inside river channel
                if self._is_river(bx, by, margin_m=45.0):
                    continue

                b_type = np.random.choice(spec["types"])
                w = float(np.random.uniform(spec["w_range"][0], spec["w_range"][1]))
                d = float(np.random.uniform(spec["w_range"][0], spec["w_range"][1]) * np.random.uniform(0.8, 1.4))
                h = float(np.random.uniform(spec["h_min"], spec["h_max"]))
                color = np.random.choice(spec["colors"])
                elev = self._get_elev(bx, by)
                stories = int(max(1, round(h / 3.5)))

                # Create rectangular footprint polygon
                hw, hd = w / 2.0, d / 2.0
                poly = [
                    [bx - hw, by - hd],
                    [bx + hw, by - hd],
                    [bx + hw, by + hd],
                    [bx - hw, by + hd],
                    [bx - hw, by - hd]
                ]

                has_helipad = (h > 90.0 and np.random.rand() < 0.4) or (b_type == "hospital")
                roof_type = "flat" if h > 30.0 else np.random.choice(["flat", "gabled", "hipped"])

                feat = {
                    "type": "Feature",
                    "properties": {
                        "id": f"BLDG-{b_idx:05d}",
                        "name": f"{spec['name']} #{placed + 1}",
                        "district_id": spec["district_id"],
                        "building_type": b_type,
                        "height_m": round(h, 1),
                        "stories": stories,
                        "elevation_m": round(elev, 2),
                        "width_m": round(w, 1),
                        "depth_m": round(d, 1),
                        "center": [round(bx, 1), round(by, 1)],
                        "facade_color": color,
                        "roof_type": roof_type,
                        "has_helipad": has_helipad,
                        "occupancy": int(stories * (w * d) / 25.0)
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [poly]
                    }
                }
                features.append(feat)
                b_idx += 1
                placed += 1

        # Landmark campus buildings (Hospitals, Substations, Pump Stations)
        landmarks = [
            # St. Jude Trauma Hospital Complex
            {
                "id": "CAMPUS-HOSP-NORTH",
                "name": "St. Jude Trauma Center Main Pavilion",
                "district_id": "DIST-PINE-HEIGHTS",
                "type": "hospital",
                "cx": 1800.0, "cy": 4800.0, "w": 90.0, "d": 75.0, "h": 48.0,
                "color": "#ef4444", "helipad": True
            },
            # Metro General Hospital
            {
                "id": "CAMPUS-HOSP-CENTRAL",
                "name": "MetroVerdia Central Hospital Tower",
                "district_id": "DIST-MIDTOWN",
                "type": "hospital",
                "cx": 4100.0, "cy": 2200.0, "w": 85.0, "d": 70.0, "h": 55.0,
                "color": "#ef4444", "helipad": True
            },
            # Verdia City Hall & Spire
            {
                "id": "CAMPUS-CITY-HALL",
                "name": "MetroVerdia Civic Spire & Hall",
                "district_id": "DIST-DOWNTOWN",
                "type": "civic_landmark",
                "cx": 3200.0, "cy": 3800.0, "w": 65.0, "d": 65.0, "h": 185.0,
                "color": "#06b6d4", "helipad": True
            },
            # Central Financial Exchange Tower
            {
                "id": "CAMPUS-FINANCE-TOWER",
                "name": "Verdia International Trade Tower",
                "district_id": "DIST-DOWNTOWN",
                "type": "skyscraper",
                "cx": 3450.0, "cy": 3550.0, "w": 75.0, "d": 60.0, "h": 210.0,
                "color": "#38bdf8", "helipad": True
            }
        ]

        for lm in landmarks:
            elev = self._get_elev(lm["cx"], lm["cy"])
            hw, hd = lm["w"] / 2.0, lm["d"] / 2.0
            poly = [
                [lm["cx"] - hw, lm["cy"] - hd],
                [lm["cx"] + hw, lm["cy"] - hd],
                [lm["cx"] + hw, lm["cy"] + hd],
                [lm["cx"] - hw, lm["cy"] + hd],
                [lm["cx"] - hw, lm["cy"] - hd]
            ]
            feat = {
                "type": "Feature",
                "properties": {
                    "id": lm["id"],
                    "name": lm["name"],
                    "district_id": lm["district_id"],
                    "building_type": lm["type"],
                    "height_m": lm["h"],
                    "stories": int(round(lm["h"] / 3.5)),
                    "elevation_m": round(elev, 2),
                    "width_m": lm["w"],
                    "depth_m": lm["d"],
                    "center": [lm["cx"], lm["cy"]],
                    "facade_color": lm["color"],
                    "roof_type": "flat",
                    "has_helipad": lm["helipad"],
                    "is_landmark": True,
                    "occupancy": 1500
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [poly]
                }
            }
            features.append(feat)

        fc = {"type": "FeatureCollection", "features": features}
        out_path = self.raw_gis_dir / "urban_buildings.geojson"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)

        print(f"Generated {len(features)} 3D urban buildings across 6 districts in {out_path}.")
        return fc


if __name__ == "__main__":
    gen = UrbanMorphologyGenerator()
    gen.generate_districts()
    gen.generate_buildings()

