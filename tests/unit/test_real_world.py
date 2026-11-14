"""
Unit and integration tests for Phase 10 (Real-World Geographic Ingestion & Calibration):
- Real DEM ingestion & nodata interpolation
- OpenStreetMap road network geometry & elevation snapping
- LULC land-cover parameter derivation
- Scientific calibration protocol & data provenance certification
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from cityverse_sim.real_world.ingestion import RealWorldGeoIngestor
from cityverse_sim.real_world.calibration import ModelCalibrationProtocol, ModelProvenanceTier
from backend.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_real_world_dem_ingestion_and_nodata():
    ingestor = RealWorldGeoIngestor(target_resolution_m=50.0)
    
    # 20x20 elevation matrix with nodata (-9999.0) in corner
    raw_dem = np.full((20, 20), 25.0, dtype=np.float32)
    raw_dem[0:2, 0:2] = -9999.0
    
    grid = ingestor.ingest_dem_raster(
        elevation_matrix=raw_dem,
        extent_x_m=1000.0,
        extent_y_m=1000.0,
        nodata_val=-9999.0
    )
    
    assert grid.rows == 20
    assert grid.cols == 20
    assert not (grid.dem == -9999.0).any(), "Nodata values must be safely filled."
    assert grid.dem[0, 0] >= 0.0


def test_osm_road_ingestion_and_snapping():
    ingestor = RealWorldGeoIngestor(target_resolution_m=50.0)
    
    # Setup simple 20x20 terrain
    dem = np.linspace(10.0, 30.0, 400).reshape((20, 20)).astype(np.float32)
    grid = ingestor.ingest_dem_raster(dem, extent_x_m=1000.0, extent_y_m=1000.0)
    
    # Sample OpenStreetMap LineString features
    osm_features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[100.0, 500.0], [500.0, 500.0], [900.0, 500.0]]
            },
            "properties": {
                "highway": "primary",
                "bridge": "yes"
            }
        }
    ]
    
    road_net = ingestor.ingest_osm_roads(grid, osm_features)
    assert road_net.graph.number_of_nodes() == 3
    assert road_net.graph.number_of_edges() == 4 # 2 bidirectional segments
    
    # Verify elevation was snapped from terrain
    for _, data in road_net.graph.nodes(data=True):
        assert data["z"] >= 10.0
        
    for _, _, data in road_net.graph.edges(data=True):
        assert data["road_class"] == "highway"
        assert data["is_bridge"] is True


def test_lulc_derivation():
    ingestor = RealWorldGeoIngestor(target_resolution_m=50.0)
    grid = ingestor.ingest_dem_raster(np.zeros((10, 10)), extent_x_m=500.0, extent_y_m=500.0)
    
    # LULC map: class 50 (urban) in top half, class 10 (forest) in bottom half
    lulc = np.full((10, 10), 10, dtype=np.int32)
    lulc[0:5, :] = 50
    
    ingestor.derive_land_cover_properties(grid, lulc)
    
    # Urban top half must have high impervious fraction (0.85)
    assert (grid.impervious_fraction[0:5, :] == 0.85).all()
    assert (grid.manning_n[0:5, :] == 0.015).all()
    
    # Forest bottom half must have low impervious fraction (0.10)
    assert (grid.impervious_fraction[5:10, :] == 0.10).all()
    assert (grid.manning_n[5:10, :] == 0.060).all()


def test_model_calibration_and_provenance():
    protocol = ModelCalibrationProtocol(city_name="MetroVerdia")
    assert protocol.tier == ModelProvenanceTier.SYNTHETIC_SIMULATION
    
    # 1. Ingest real-world geography
    protocol.set_real_world_uncalibrated("Houston_Sector_4", dem_source="USGS_3DEP", osm_source="OpenStreetMap_2026")
    assert protocol.tier == ModelProvenanceTier.REAL_WORLD_UNCALIBRATED
    manifest_uncal = protocol.get_provenance_manifest()
    assert manifest_uncal["is_real_world_calibrated"] is False
    
    # 2. Historical event validation (Hurricane Harvey benchmark)
    simulated_h = np.zeros((50, 50), dtype=np.float32)
    simulated_h[15:35, 15:35] = 0.40 # 20x20 flooded area
    
    # Perfect match observation
    observed_mask = np.zeros((50, 50), dtype=bool)
    observed_mask[15:35, 15:35] = True
    
    res = protocol.calibrate_against_historical_event(
        event_name="Hurricane_Harvey_2017",
        observed_flood_extent_mask=observed_mask,
        simulated_depth_grid=simulated_h,
        depth_threshold_m=0.15
    )
    
    assert res["is_certified"] is True
    assert protocol.tier == ModelProvenanceTier.REAL_WORLD_CALIBRATED_BENCHMARK
    manifest_cal = protocol.get_provenance_manifest()
    assert manifest_cal["is_real_world_calibrated"] is True
    assert manifest_cal["calibration_event"] == "Hurricane_Harvey_2017"


def test_api_provenance_endpoint(client):
    res = client.get("/api/real_world/provenance")
    assert res.status_code == 200
    data = res.json()
    assert "provenance_tier" in data
    assert "disclaimer" in data

