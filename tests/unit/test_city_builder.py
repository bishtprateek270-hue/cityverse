"""
Unit tests for MetroVerdia procedural city generation (Phase 1).
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path

from cityverse_sim.city.grid import CityGrid
from cityverse_sim.city.network import RoadNetwork, DrainageNetwork, FacilityRegistry
from cityverse_sim.city.city_builder import MetroVerdiaCity, build_and_export_metroverdia


@pytest.fixture
def city_instance():
    city = MetroVerdiaCity("configs/city_spec.yaml")
    city.build(seed=42)
    return city


def test_grid_initialization_and_transforms():
    grid = CityGrid(extent_x_m=6000.0, extent_y_m=6000.0, resolution_m=50.0)
    assert grid.rows == 120
    assert grid.cols == 120
    
    # Test coordinate transforms round-trip
    x_test, y_test = 2525.0, 3475.0
    r, c = grid.coord_to_cell(x_test, y_test)
    x_rec, y_rec = grid.cell_to_coord(r, c)
    assert abs(x_rec - 2525.0) <= 25.0
    assert abs(y_rec - 3475.0) <= 25.0


def test_terrain_and_slopes(city_instance):
    grid = city_instance.grid
    
    # Elevation bounds check
    assert grid.dem.shape == (120, 120)
    assert 12.0 <= grid.dem.min() <= 15.0
    assert 55.0 <= grid.dem.max() <= 68.0
    
    # Check river cells
    assert grid.river_mask.sum() > 50
    # River bed should be lower than surrounding terrain
    river_mean_elev = grid.dem[grid.river_mask].mean()
    overall_mean_elev = grid.dem.mean()
    assert river_mean_elev < overall_mean_elev
    
    # Check slope calculation
    dz_dx, dz_dy, slope_mag = grid.calculate_slopes()
    assert slope_mag.shape == (120, 120)
    assert np.all(slope_mag >= 0.0)
    assert slope_mag.max() > 0.01


def test_road_network_properties(city_instance):
    roads = city_instance.roads.graph
    assert roads.number_of_nodes() > 100
    assert roads.number_of_edges() > 300
    
    # Check bridge detection
    bridges = [d for _, _, d in roads.edges(data=True) if d.get("is_bridge")]
    assert len(bridges) > 0, "Road network should have bridge links crossing the river."
    
    # Check highway exists
    hwys = [d for _, _, d in roads.edges(data=True) if d.get("road_class") == "highway"]
    assert len(hwys) > 0
    
    # Every road edge must have positive length and speed
    for _, _, d in roads.edges(data=True):
        assert d["length_m"] > 0
        assert d["free_flow_speed_mps"] > 0
        assert d["capacity_veh_per_hr"] > 0


def test_drainage_network_slopes(city_instance):
    drainage = city_instance.drainage.graph
    assert drainage.number_of_nodes() > 50
    assert drainage.number_of_edges() > 50
    
    outfalls = [n for n, d in drainage.nodes(data=True) if d.get("node_type") == "outfall"]
    assert len(outfalls) == 3
    
    # Ensure conduits have valid slopes and capacities
    for _, _, d in drainage.edges(data=True):
        assert d["slope"] > 0
        assert d["diameter_m"] >= 0.5
        assert d["capacity_m3_s"] > 0.1


def test_facility_registration(city_instance):
    facilities = city_instance.facilities
    
    hospitals = facilities.get_by_category("hospitals")
    assert len(hospitals) == 2
    for h in hospitals:
        assert h["nearest_road_node"] in city_instance.roads.graph.nodes
        assert h["capacity_beds"] > 0
        
    ems = facilities.get_by_category("ems_stations")
    assert len(ems) == 4
    
    subs = facilities.get_by_category("electrical_substations")
    assert len(subs) == 3
    
    pumps = facilities.get_by_category("pumping_stations")
    assert len(pumps) == 2


def test_gis_export_and_file_integrity(tmp_path):
    city = MetroVerdiaCity("configs/city_spec.yaml")
    city.build(seed=42)
    summary = city.export_gis_data(output_dir=str(tmp_path))
    
    # Check files exist
    assert (tmp_path / "dem_elevation.npy").exists()
    assert (tmp_path / "manning_n.npy").exists()
    assert (tmp_path / "impervious_fraction.npy").exists()
    assert (tmp_path / "road_network.geojson").exists()
    assert (tmp_path / "drainage_conduits.geojson").exists()
    assert (tmp_path / "critical_facilities.geojson").exists()
    assert (tmp_path / "city_summary.json").exists()
    
    # Validate GeoJSON parsing
    with open(tmp_path / "road_network.geojson", "r", encoding="utf-8") as f:
        road_data = json.load(f)
        assert road_data["type"] == "FeatureCollection"
        assert len(road_data["features"]) > 0

