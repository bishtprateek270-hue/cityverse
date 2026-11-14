"""
Integration tests for FastAPI REST Endpoints (Phase 8):
- Health check & city GIS layers
- NLP Scenario prompt parsing
- Simulation start, step, and live interventions
- Counterfactual branch forking and comparison API
"""

import pytest
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_and_info(client):
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "online"
    
    res_info = client.get("/api/city/info")
    assert res_info.status_code == 200
    assert res_info.json()["city_name"] == "MetroVerdia"


def test_api_gis_layers(client):
    res_terrain = client.get("/api/city/terrain")
    assert res_terrain.status_code == 200
    data = res_terrain.json()
    assert data["rows"] == 120
    assert data["cols"] == 120
    
    res_roads = client.get("/api/city/roads")
    assert res_roads.status_code == 200
    assert res_roads.json()["type"] == "FeatureCollection"
    
    res_fac = client.get("/api/city/facilities")
    assert res_fac.status_code == 200
    assert res_fac.json()["type"] == "FeatureCollection"

    res_bldg = client.get("/api/city/buildings")
    assert res_bldg.status_code == 200
    assert res_bldg.json()["type"] == "FeatureCollection"
    assert len(res_bldg.json()["features"]) >= 1000

    res_dist = client.get("/api/city/districts")
    assert res_dist.status_code == 200
    assert res_dist.json()["type"] == "FeatureCollection"
    assert len(res_dist.json()["features"]) >= 5


def test_api_nlp_scenario_parser(client):
    prompt = "Simulate 120 mm/hr rainfall over 3 hours with pump 1 failure and drainage capacity reduced by 40%"
    res = client.post("/api/scenario/parse", json={"prompt": prompt})
    assert res.status_code == 200
    data = res.json()
    
    assert data["rainfall_mm_hr"] == 120.0
    assert data["duration_hours"] == 3.0
    assert data["drainage_capacity_factor"] == 0.6
    assert data["pump_overrides"]["PUMP-1"] == 0.0
    assert len(data["recognized_entities"]) >= 3


def test_api_simulation_lifecycle_and_stepping(client):
    # 1. Start simulation
    res_start = client.post("/api/simulation/start", json={
        "rainfall_mm_hr": 80.0,
        "duration_hours": 2.0,
        "drainage_capacity_factor": 0.9,
        "engine_mode": "world_model"
    })
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "started"
    
    # 2. Step forward with World Model
    res_step = client.post("/api/simulation/step", json={
        "branch_id": "baseline",
        "engine_mode": "world_model",
        "dt_s": 300.0
    })
    assert res_step.status_code == 200
    step_data = res_step.json()
    assert step_data["time_s"] == 300.0
    assert "surface_depth_grid" in step_data
    assert "snapshot" in step_data
    
    # 3. Inject intervention
    res_interv = client.post("/api/simulation/intervention", json={
        "branch_id": "baseline",
        "rainfall_mm_hr": 120.0,
        "drainage_capacity_factor": 0.5
    })
    assert res_interv.status_code == 200
    assert res_interv.json()["status"] == "injected"


def test_api_counterfactual_branch_and_compare(client):
    # Start fresh simulation
    client.post("/api/simulation/start", json={"rainfall_mm_hr": 90.0, "duration_hours": 2.0, "drainage_capacity_factor": 1.0})
    
    # Advance 1 step
    client.post("/api/simulation/step", json={"branch_id": "baseline", "engine_mode": "world_model", "dt_s": 300.0})
    
    # Fork Branch B
    res_fork = client.post("/api/branch/fork", json={
        "source_branch_id": "baseline",
        "new_branch_id": "branch_emergency_pumps",
        "intervention": {"pump_overrides": {"PUMP-1": 1.0, "PUMP-2": 1.0}, "drainage_capacity_factor": 1.0}
    })
    assert res_fork.status_code == 200
    assert res_fork.json()["status"] == "forked"
    assert "branch_emergency_pumps" in res_fork.json()["active_branches"]
    
    # Step Branch B
    client.post("/api/simulation/step", json={"branch_id": "branch_emergency_pumps", "engine_mode": "world_model", "dt_s": 300.0})
    
    # Compare Branch baseline vs Branch B
    res_comp = client.post("/api/branch/compare", json={
        "branch_a_id": "baseline",
        "branch_b_id": "branch_emergency_pumps"
    })
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert "delta_flooded_area_km2" in comp_data
    assert "is_branch_b_better" in comp_data

