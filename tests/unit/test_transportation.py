"""
Unit and dynamic integration tests for Phase 3:
- Pregnolato speed reduction curve
- BPR volume-delay congestion
- Dynamic vehicle rerouting around flooded streets
- Coupled Hydro-Traffic cascade: storm rainfall causes traffic delays & road closures
"""

import pytest
import numpy as np

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.hydraulics.coupled_solver import CoupledHydraulicsEngine
from cityverse_sim.transportation.traffic_graph import DynamicRoadNetwork
from cityverse_sim.transportation.congestion import BPRCongestionModel
from cityverse_sim.transportation.routing import TrafficRouter


@pytest.fixture
def metro_city():
    city = MetroVerdiaCity("configs/city_spec.yaml")
    city.build(seed=42)
    return city


def test_pregnolato_speed_degradation(metro_city):
    dyn_roads = DynamicRoadNetwork(metro_city.roads, metro_city.grid)
    
    # 1. Zero water (dry): full speed, 100% passable
    dry_surface = np.zeros((120, 120), dtype=np.float32)
    dyn_roads.update_flood_conditions(dry_surface)
    summary = dyn_roads.get_summary_metrics()
    assert summary["closed_edges_passenger"] == 0
    assert summary["closed_edges_emergency"] == 0
    
    # Check speed on first edge
    u, v, data = list(dyn_roads.graph.edges(data=True))[0]
    assert data["flood_speed_mps"] == data["free_flow_speed_mps"]
    
    # 2. Introduce 0.18m water in the urban core
    wet_surface = np.zeros((120, 120), dtype=np.float32)
    wet_surface[40:80, 40:80] = 0.18 # 18 cm of water
    dyn_roads.update_flood_conditions(wet_surface)
    summary_wet = dyn_roads.get_summary_metrics()
    
    # 18 cm is > 15 cm passenger limit -> some passenger roads must close
    assert summary_wet["closed_edges_passenger"] > 0
    # 18 cm is < 30 cm emergency limit -> emergency roads remain passable
    assert summary_wet["closed_edges_emergency"] < summary_wet["closed_edges_passenger"]


def test_bpr_congestion_delay():
    bpr = BPRCongestionModel(alpha_bpr=0.15, beta_bpr=4.0)
    length = 1000.0 # 1 km
    speed = 20.0 # 20 m/s = 72 km/h
    cap = 1000.0 # veh/hr
    
    # Zero traffic: T0 = 1000 / 20 = 50s
    t_zero = bpr.compute_edge_travel_time(length, speed, volume_veh_hr=0.0, capacity_veh_hr=cap)
    assert abs(t_zero - 50.0) < 1e-3
    
    # Congested traffic (V = 1.5 * C): T = 50 * [1 + 0.15 * (1.5^4)] = 50 * [1 + 0.15 * 5.0625] = 87.96s
    t_congested = bpr.compute_edge_travel_time(length, speed, volume_veh_hr=1500.0, capacity_veh_hr=cap)
    assert t_congested > t_zero * 1.5
    
    # Closed edge
    t_closed = bpr.compute_edge_travel_time(length, flood_speed_mps=0.0, volume_veh_hr=0.0, capacity_veh_hr=cap, is_passable=False)
    assert t_closed >= 1e8


def test_dynamic_rerouting_around_flooded_road(metro_city):
    dyn_roads = DynamicRoadNetwork(metro_city.roads, metro_city.grid)
    router = TrafficRouter(dyn_roads)
    
    # Update with dry condition
    dry_surface = np.zeros((120, 120), dtype=np.float32)
    dyn_roads.update_flood_conditions(dry_surface)
    bpr = BPRCongestionModel()
    bpr.update_network_travel_times(dyn_roads.graph, {}, vehicle_type="passenger")
    
    # Pick an active OD trip
    orig, dest, _ = router.od_trips[0]
    dry_path_res = router.compute_shortest_path(orig, dest, vehicle_type="passenger")
    assert dry_path_res["is_reachable"] is True
    original_path = dry_path_res["path_nodes"]
    original_time = dry_path_res["travel_time_s"]
    
    # Submerge the first intermediate road segment on this path
    u_cut = original_path[1]
    v_cut = original_path[2]
    
    # Find cell for u_cut and flood it with 0.50m water
    u_node = dyn_roads.graph.nodes[u_cut]
    r, c = metro_city.grid.coord_to_cell(u_node["x"], u_node["y"])
    
    flooded_surface = np.zeros((120, 120), dtype=np.float32)
    # Flood around this intersection
    flooded_surface[max(0, r-2):min(120, r+3), max(0, c-2):min(120, c+3)] = 0.50
    
    dyn_roads.update_flood_conditions(flooded_surface)
    bpr.update_network_travel_times(dyn_roads.graph, {}, vehicle_type="passenger")
    
    # Compute path again
    detour_res = router.compute_shortest_path(orig, dest, vehicle_type="passenger")
    if detour_res["is_reachable"]:
        detour_path = detour_res["path_nodes"]
        # Detour path should not traverse the flooded edge (u_cut, v_cut)
        edges_used = detour_res["path_edges"]
        assert (u_cut, v_cut) not in edges_used
        assert detour_res["travel_time_s"] >= original_time


def test_storm_traffic_coupling(metro_city):
    """
    Coupled Integration Test:
    Simulates a heavy 40-minute storm, feeds resulting flood depth into traffic router,
    and measures road closures and congestion impact.
    """
    hydro_engine = CoupledHydraulicsEngine(metro_city.grid, metro_city.drainage, metro_city.facilities)
    dyn_roads = DynamicRoadNetwork(metro_city.roads, metro_city.grid)
    router = TrafficRouter(dyn_roads)
    
    # Baseline dry traffic assignment
    dry_surface = np.zeros((120, 120), dtype=np.float32)
    dry_metrics = router.assign_traffic_and_update(dry_surface)
    assert dry_metrics["closed_edges_passenger"] == 0
    assert dry_metrics["stranded_commute_trips"] == 0
    
    # Run 40 minutes of 90 mm/hr storm
    dt = 120.0
    for _ in range(20):
        hydro_engine.step(rainfall_rate_mm_hr=90.0, dt_s=dt, drainage_capacity_factor=0.6)
        
    flooded_depth = hydro_engine.overland.h
    assert np.max(flooded_depth) > 0.15 # Flooding occurred
    
    # Run coupled traffic assignment under flood
    storm_metrics = router.assign_traffic_and_update(flooded_depth)
    
    print("\nCoupled Storm Traffic Impact:")
    print(f"  - Closed Edges (Passenger): {storm_metrics['closed_edges_passenger']} ({storm_metrics['passenger_closure_pct']:.1f}%)")
    print(f"  - Closed Edges (Emergency): {storm_metrics['closed_edges_emergency']} ({storm_metrics['emergency_closure_pct']:.1f}%)")
    print(f"  - Stranded Commute Trips:   {storm_metrics['stranded_commute_trips']} / {storm_metrics['total_commute_trips']}")
    print(f"  - Baseline Avg Commute:     {dry_metrics['avg_commute_time_min']:.1f} min")
    print(f"  - Storm Avg Commute:        {storm_metrics['avg_commute_time_min']:.1f} min")
    
    assert storm_metrics["closed_edges_passenger"] > 0, "Heavy storm must inundate low-lying road links."
    assert (storm_metrics["avg_commute_time_min"] > dry_metrics["avg_commute_time_min"] or
            storm_metrics["stranded_commute_trips"] > 0), "Flood must disrupt transportation."
