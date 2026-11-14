"""
Unit and dynamic counterfactual tests for Phase 7:
- Mid-simulation action injection (rain, pumps, roadblocks)
- Counterfactual scenario forking & state divergence
- Quantitative branch diffing (Flooded Area, EMS delays, Substations)
- Neural World Model fast-forward rollouts under interventions
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.interventions.state_manager import CityStateManager
from cityverse_sim.interventions.branch_engine import CounterfactualBranchEngine
from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel


@pytest.fixture
def metro_city():
    city = MetroVerdiaCity("configs/city_spec.yaml")
    city.build(seed=42)
    return city


def test_state_manager_mid_simulation_injection(metro_city):
    mgr = CityStateManager(city=metro_city)
    
    # 1. Step 1: 0 rain -> dry
    mgr.step_physics(dt_s=60.0)
    assert mgr.hydro.overland.h.max() == 0.0
    
    # 2. Mid-simulation intervention: inject heavy rain at t=60s
    mgr.inject_intervention({"rainfall_mm_hr": 80.0, "drainage_capacity_factor": 0.5})
    assert mgr.active_rainfall_mm_hr == 80.0
    assert mgr.drainage_capacity_factor == 0.5
    
    # 3. Advance physics
    snap = mgr.step_physics(dt_s=120.0)
    assert snap["max_water_depth_m"] > 0.001
    
    # 4. Inject roadblock on first edge
    u, v = list(metro_city.roads.graph.edges())[0]
    mgr.inject_intervention({"block_road": (u, v)})
    assert (u, v) in mgr.manual_roadblocks
    
    # Step again and verify road is marked impassable
    mgr.step_physics(dt_s=60.0)
    assert mgr.dyn_roads.graph[u][v]["is_passable_passenger"] is False


def test_counterfactual_branch_divergence(metro_city):
    """
    Counterfactual Test:
    Starts with a storm. At t=15 min, forks into:
      - Branch A: Status Quo (80% drainage blockage, pumps offline)
      - Branch B: Emergency Mitigation (pumps running 100%, drainage unblocked)
    Asserts that Branch B significantly reduces flood extent and improves response.
    """
    root = CityStateManager(city=metro_city)
    # Baseline rain 85 mm/hr
    root.inject_intervention({"rainfall_mm_hr": 85.0})
    
    # Run initial 15 minutes (900s)
    for _ in range(15):
        root.step_physics(dt_s=60.0)
        
    engine = CounterfactualBranchEngine(root)
    
    # Fork Branch A: Failure / unmitigated
    engine.fork_branch(
        new_branch_id="unmitigated",
        source_branch_id="baseline",
        intervention={
            "drainage_capacity_factor": 0.2, # Heavily choked
            "pump_overrides": {"PUMP-1": 0.0, "PUMP-2": 0.0} # Offline
        }
    )
    
    # Fork Branch B: Active mitigation
    engine.fork_branch(
        new_branch_id="mitigated",
        source_branch_id="baseline",
        intervention={
            "drainage_capacity_factor": 1.0, # Clean
            "pump_overrides": {"PUMP-1": 1.0, "PUMP-2": 1.0} # Full power
        }
    )
    
    # Rollout both branches forward for 20 minutes (1200s)
    rollouts = engine.rollout_branches(["unmitigated", "mitigated"], duration_s=1200.0, dt_s=120.0)
    assert len(rollouts["unmitigated"]) == 10
    assert len(rollouts["mitigated"]) == 10
    
    # Compare outcomes
    diff = engine.compare_branches("unmitigated", "mitigated")
    
    print("\nCounterfactual Branch Comparison (Unmitigated vs Mitigated):")
    print(f"  - Delta Flooded Area (>15cm): {diff['delta_flooded_area_km2']:.3f} km2")
    print(f"  - Delta EMS Response Time:    {diff['delta_avg_ems_time_min']:.2f} min")
    print(f"  - Mitigated Branch Effective: {diff['is_branch_b_better']}")
    
    # Branch B must accumulate less flood water than Branch A
    vol_unmitigated = np.sum(engine.branches["unmitigated"].hydro.overland.h)
    vol_mitigated = np.sum(engine.branches["mitigated"].hydro.overland.h)
    assert vol_mitigated < vol_unmitigated, "Mitigated branch must reduce surface water volume."
    assert diff["is_branch_b_better"] is True


def test_neural_world_model_fast_counterfactual(metro_city):
    ckpt_path = Path("models/checkpoints/world_model_best.pt")
    if not ckpt_path.exists():
        pytest.skip("Trained checkpoint not found.")
        
    model = UrbanWorldModel(in_grid_channels=6, base_channels=24, scalar_dim=11, action_dim=4, num_road_edges=464)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    
    mgr = CityStateManager(city=metro_city, world_model=model)
    mgr.inject_intervention({"rainfall_mm_hr": 90.0, "drainage_capacity_factor": 0.8})
    
    # Step forward 3 times using fast neural inference
    import time
    t0 = time.perf_counter()
    snap1 = mgr.step_world_model(dt_s=300.0)
    snap2 = mgr.step_world_model(dt_s=300.0)
    snap3 = mgr.step_world_model(dt_s=300.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    assert snap3["time_s"] == 900.0
    assert snap3["max_water_depth_m"] >= 0.0
    assert elapsed_ms < 150.0, "3 neural rollout steps should take <150ms on CPU."

