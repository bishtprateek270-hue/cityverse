"""
Unit and dynamic cascading tests for Phase 4:
- Electrical substation flooding & breaker trip
- Cascading power outage shutting down stormwater pumping stations
- Emergency medical dispatch routing & latency inflation under flood conditions
"""

import pytest
import numpy as np

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.hydraulics.coupled_solver import CoupledHydraulicsEngine
from cityverse_sim.transportation.traffic_graph import DynamicRoadNetwork
from cityverse_sim.transportation.congestion import BPRCongestionModel
from cityverse_sim.transportation.routing import TrafficRouter
from cityverse_sim.infrastructure.electrical_grid import ElectricalGridManager
from cityverse_sim.emergency.dispatch import EmergencyDispatcher


@pytest.fixture
def metro_city():
    city = MetroVerdiaCity("configs/city_spec.yaml")
    city.build(seed=42)
    return city


def test_electrical_grid_substation_trip(metro_city):
    power_mgr = ElectricalGridManager(metro_city.grid, metro_city.facilities)
    
    # 1. Dry conditions
    dry_surface = np.zeros((120, 120), dtype=np.float32)
    dry_status = power_mgr.update_grid_status(dry_surface)
    assert dry_status["operational_substations"] == 3
    assert len(dry_status["tripped_substations"]) == 0
    assert len(dry_status["disabled_pumps"]) == 0
    
    # 2. Submerge Substation Gamma yard (x=4500, y=1600) with 0.35m of water
    r, c = metro_city.grid.coord_to_cell(4500.0, 1600.0)
    wet_surface = np.zeros((120, 120), dtype=np.float32)
    wet_surface[r-1:r+2, c-1:c+2] = 0.35
    
    wet_status = power_mgr.update_grid_status(wet_surface)
    assert "SUB-GAMMA" in wet_status["tripped_substations"]
    assert "PUMP-1" in wet_status["disabled_pumps"] # Pump 1 loses power from Substation Gamma!
    assert wet_status["operational_substations"] == 2


def test_cascading_pump_failure_flood_worsening(metro_city):
    """
    Causal Cascade Test:
    Demonstrates that tripping Substation Gamma disables Pump 1, causing
    significantly higher water accumulation in the alluvial basin.
    """
    hydro_pump_active = CoupledHydraulicsEngine(metro_city.grid, metro_city.drainage, metro_city.facilities)
    hydro_pump_disabled = CoupledHydraulicsEngine(metro_city.grid, metro_city.drainage, metro_city.facilities)
    
    dt = 120.0
    rain = 70.0 # mm/hr
    
    # Simulate 30 minutes
    for _ in range(15):
        # Active pump runs normally
        hydro_pump_active.step(rain, dt, pump_override={"PUMP-1": 1.0})
        # Disabled pump is throttled to 0 (power failure)
        hydro_pump_disabled.step(rain, dt, pump_override={"PUMP-1": 0.0})
        
    vol_active = np.sum(hydro_pump_active.overland.h)
    vol_disabled = np.sum(hydro_pump_disabled.overland.h)
    
    # Without the pump running, more surface water remains trapped in the basin
    assert vol_disabled > vol_active, "Pump station failure must increase surface water retention."


def test_emergency_dispatch_latency_under_flood(metro_city):
    """
    Emergency Cascade Test:
    Evaluates response times for first responders under dry vs flooded conditions.
    """
    dyn_roads = DynamicRoadNetwork(metro_city.roads, metro_city.grid)
    bpr = BPRCongestionModel()
    router = TrafficRouter(dyn_roads, bpr)
    dispatcher = EmergencyDispatcher(metro_city.facilities, router)
    
    # 1. Baseline Dry Assessment
    dry_surface = np.zeros((120, 120), dtype=np.float32)
    dyn_roads.update_flood_conditions(dry_surface)
    bpr.update_network_travel_times(dyn_roads.graph, {}, vehicle_type="emergency")
    
    # Sample 30 random city intersections
    sample_nodes = list(dyn_roads.graph.nodes())[:30]
    dry_cov = dispatcher.evaluate_emergency_coverage(sample_nodes)
    
    assert dry_cov["coverage_pct"] == 100.0, "All nodes must be reachable under dry conditions."
    assert dry_cov["avg_response_time_min"] < 15.0, "Baseline emergency response should be under 15 minutes."
    
    # 2. Severe Inundation Assessment: flood central river corridor and low basin with 0.40m water
    wet_surface = np.zeros((120, 120), dtype=np.float32)
    # Flood river corridor and low-lying quadrant
    wet_surface[metro_city.grid.river_mask] = 0.45
    wet_surface[70:110, 60:110] = 0.40
    
    dyn_roads.update_flood_conditions(wet_surface)
    bpr.update_network_travel_times(dyn_roads.graph, {}, vehicle_type="emergency")
    
    wet_cov = dispatcher.evaluate_emergency_coverage(sample_nodes)
    
    print("\nEmergency Response Cascade Impact:")
    print(f"  - Dry Coverage:          {dry_cov['coverage_pct']:.1f}% (Avg: {dry_cov['avg_response_time_min']:.2f} min)")
    print(f"  - Flooded Coverage:      {wet_cov['coverage_pct']:.1f}% (Avg: {wet_cov['avg_response_time_min']:.2f} min)")
    print(f"  - Isolated Nodes:        {wet_cov['isolated_stranded_nodes']} / {wet_cov['total_assessed_nodes']}")
    print(f"  - Golden Hour Violation: {wet_cov['golden_hour_violation_pct']:.1f}%")
    
    # Flood must degrade emergency service
    assert (wet_cov["avg_response_time_min"] > dry_cov["avg_response_time_min"] or
            wet_cov["isolated_stranded_nodes"] > 0)

