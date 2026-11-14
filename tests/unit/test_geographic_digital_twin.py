"""
tests/unit/test_geographic_digital_twin.py
==========================================
Unit & Integration tests for the Geographic Urban Digital Twin:
- Analytical hillshade layer integrity
- Urban blocks & building footprints
- Batch scenario trajectory rollout & caching
- Timeline frame scrubber retrieval
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app
from backend.timeline_cache import TimelineTrajectoryCache
from cityverse_sim.city.geo_fabric import GeoFabricGenerator


@pytest.fixture
def client():
    return TestClient(app)


def test_hillshade_computation():
    gen = GeoFabricGenerator()
    hs = gen.generate_hillshade(azimuth_deg=315.0, altitude_deg=45.0)
    assert hs.shape == (120, 120)
    assert 0.0 <= float(hs.min())
    assert float(hs.max()) <= 1.0


def test_urban_blocks_and_footprints():
    gen = GeoFabricGenerator()
    blocks, footprints = gen.generate_urban_blocks_and_footprints()
    assert len(blocks["features"]) >= 30
    assert len(footprints["features"]) >= 100
    
    # Check properties
    first_bldg = footprints["features"][0]["properties"]
    assert "stories" in first_bldg
    assert "height_m" in first_bldg
    assert "zoning" in first_bldg


def test_api_geographic_endpoints(client):
    res_hs = client.get("/api/city/hillshade")
    assert res_hs.status_code == 200
    assert len(res_hs.json()["hillshade_grid"]) == 120

    res_blocks = client.get("/api/city/blocks")
    assert res_blocks.status_code == 200
    assert res_blocks.json()["type"] == "FeatureCollection"

    res_fp = client.get("/api/city/footprints")
    assert res_fp.status_code == 200
    assert res_fp.json()["type"] == "FeatureCollection"


def test_api_batch_scenario_and_timeline_scrubber(client):
    payload = {
        "rainfall_mm_hr": 120.0,
        "duration_hours": 3.0,
        "drainage_capacity_factor": 0.6,
        "engine_mode": "world_model"
    }
    res_run = client.post("/api/simulation/run_scenario", json=payload)
    assert res_run.status_code == 200
    data = res_run.json()
    assert data["total_frames"] == 37 # 3 hours / 5 min = 36 intervals + T0
    assert len(data["time_series"]) == 37

    # Retrieve frame at T=0
    res_f0 = client.get("/api/simulation/frame?step_idx=0")
    assert res_f0.status_code == 200
    assert res_f0.json()["time_s"] == 0.0

    # Retrieve frame at Step 18 (T=90 min)
    res_f18 = client.get("/api/simulation/frame?step_idx=18")
    assert res_f18.status_code == 200
    assert res_f18.json()["time_s"] == 5400.0
    assert "surface_depth_grid" in res_f18.json()
    assert "snapshot" in res_f18.json()

