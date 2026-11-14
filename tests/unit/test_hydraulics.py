"""
Unit and hydrodynamic validation tests for Phase 2:
- Horton infiltration dynamics & supply limits
- 2D overland flow downslope physics
- Mass conservation under a severe 2-hour storm (<1.5% error)
- Drainage capacity sensitivity (blockage factor impact)
"""

import pytest
import numpy as np

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.hydraulics.infiltration import HortonInfiltration
from cityverse_sim.hydraulics.overland_2d import Overland2DSolver
from cityverse_sim.hydraulics.coupled_solver import CoupledHydraulicsEngine


@pytest.fixture
def metro_city():
    city = MetroVerdiaCity("configs/city_spec.yaml")
    city.build(seed=42)
    return city


def test_horton_infiltration():
    # Setup simple 2x2 grid
    f0 = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
    fc = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    decay = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    # Cell 0,0 is 0% impervious; Cell 0,1 is 100% impervious
    imp = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)
    
    infil_model = HortonInfiltration(f0, fc, decay, imp)
    
    # 1. Zero rain, zero depth -> zero infiltration
    surface_depth = np.zeros((2, 2), dtype=np.float32)
    infil_depth, net_rate = infil_model.step(0.0, surface_depth, dt_s=60.0)
    assert np.all(infil_depth == 0.0)
    assert np.all(net_rate == 0.0)
    
    # 2. Heavy rain: 100 mm/hr for 1 hour (dt = 3600s)
    infil_depth, net_rate = infil_model.step(100.0, surface_depth, dt_s=3600.0)
    # 100% impervious cell (0,1) must have ZERO infiltration
    assert infil_depth[0, 1] == 0.0
    # 100% impervious cell must produce max runoff (100 mm/hr -> 100 / 3.6e6 m/s)
    expected_runoff = (100.0 / 1000.0) / 3600.0
    assert abs(net_rate[0, 1] - expected_runoff) < 1e-6
    # 0% impervious cell (0,0) must infiltrate substantial water
    assert infil_depth[0, 0] > 0.015 # > 15 mm


def test_overland_downslope_flow():
    # 5x5 plane with a uniform slope from north (high) to south (low)
    dem = np.zeros((5, 5), dtype=np.float32)
    for r in range(5):
        dem[r, :] = 20.0 - r * 2.0 # 20m, 18m, 16m, 14m, 12m
    manning = np.full((5, 5), 0.030, dtype=np.float32)
    
    solver = Overland2DSolver(dem, manning, dx=50.0, dy=50.0)
    
    # Place a 1.0m slug of water at top center (row 0, col 2)
    solver.h[0, 2] = 1.0
    initial_mass = solver.h[0, 2] * solver.cell_area
    solver.cumulative_inflow_vol_m3 = initial_mass
    
    # Step forward 60 seconds
    zero_runoff = np.zeros((5, 5), dtype=np.float32)
    solver.step(dt_s=60.0, effective_runoff_rate_m_s=zero_runoff)
    
    # Water must flow downhill (southwards into row 1, col 2)
    assert solver.h[1, 2] > 0.05, "Water must flow downhill along slope gradient"
    # Downhill flow must dominate lateral dispersion
    assert solver.h[1, 2] > solver.h[0, 0]


def test_overland_mass_conservation_isolated():
    # Closed box 10x10 domain with terrain
    np.random.seed(123)
    dem = np.random.uniform(15.0, 25.0, (10, 10)).astype(np.float32)
    manning = np.full((10, 10), 0.035, dtype=np.float32)
    
    solver = Overland2DSolver(dem, manning, dx=50.0, dy=50.0)
    
    # Apply rainfall of 60 mm/hr for 10 minutes (600s)
    rain_rate = (60.0 / 1000.0) / 3600.0
    runoff = np.full((10, 10), rain_rate, dtype=np.float32)
    
    for _ in range(10): # 10 steps of 60s
        solver.step(dt_s=60.0, effective_runoff_rate_m_s=runoff)
        
    residual = solver.compute_mass_balance_residual()
    total_added = solver.cumulative_inflow_vol_m3
    
    rel_error = residual / total_added
    assert rel_error < 0.005, f"Isolated 2D solver mass balance error {rel_error:.6f} exceeds 0.5%"


def test_coupled_engine_2hr_storm_mass_conservation(metro_city):
    """
    Validation Test: 2-hour, 80 mm/hr storm across the entire 36 km2 city.
    Checks mass balance: Rain = Stored + Infiltrated + Drained + Boundary Discharge.
    Target: Relative mass conservation error < 1.5%.
    """
    engine = CoupledHydraulicsEngine(
        grid=metro_city.grid,
        drainage=metro_city.drainage,
        facilities=metro_city.facilities
    )
    engine.reset()
    
    dt_step = 120.0 # 2 minute steps
    total_duration_s = 2 * 3600.0 # 2 hours = 7200s
    n_steps = int(total_duration_s / dt_step)
    
    rain_rate_mm_hr = 80.0
    
    last_metrics = None
    for step_idx in range(n_steps):
        last_metrics = engine.step(
            rainfall_rate_mm_hr=rain_rate_mm_hr,
            dt_s=dt_step,
            drainage_capacity_factor=1.0
        )
        
    assert last_metrics is not None
    
    # Total water accounted for:
    rain_vol = last_metrics["cumulative_rain_vol_m3"]
    infil_vol = last_metrics["cumulative_infiltrated_vol_m3"]
    stored_vol = last_metrics["total_surface_water_vol_m3"]
    drain_vol = last_metrics["cumulative_drained_vol_m3"]
    outfall_vol = last_metrics["boundary_discharge_vol_m3"]
    
    total_accounted = infil_vol + stored_vol + drain_vol + outfall_vol
    abs_mass_diff = abs(rain_vol - total_accounted)
    relative_mass_error = abs_mass_diff / max(rain_vol, 1.0)
    
    print(f"\n2-Hour 80mm/hr Storm Mass Balance:")
    print(f"  - Total Rain Volume:       {rain_vol:,.1f} m3")
    print(f"  - Infiltrated into Soil:   {infil_vol:,.1f} m3 ({infil_vol/rain_vol*100:.1f}%)")
    print(f"  - Surface Stored Water:    {stored_vol:,.1f} m3 ({stored_vol/rain_vol*100:.1f}%)")
    print(f"  - Drained via Subsurface:  {drain_vol:,.1f} m3 ({drain_vol/rain_vol*100:.1f}%)")
    print(f"  - Boundary Outflow:        {outfall_vol:,.1f} m3 ({outfall_vol/rain_vol*100:.1f}%)")
    print(f"  - Relative Mass Discrepancy: {relative_mass_error*100:.4f}%")
    
    assert relative_mass_error < 0.015, f"Relative mass conservation error {relative_mass_error*100:.2f}% exceeds 1.5% target!"
    assert last_metrics["flooded_area_km2_gt15cm"] > 0.2, "Severe storm must produce detectable flooding."


def test_drainage_blockage_impact(metro_city):
    """
    Verification Test: Reducing drainage capacity (blockage) must exacerbate surface flooding.
    """
    engine_clean = CoupledHydraulicsEngine(metro_city.grid, metro_city.drainage, metro_city.facilities)
    engine_blocked = CoupledHydraulicsEngine(metro_city.grid, metro_city.drainage, metro_city.facilities)
    
    dt = 120.0
    rain = 75.0 # mm/hr
    
    # Run 30 minutes
    for _ in range(15):
        engine_clean.step(rain, dt, drainage_capacity_factor=1.0)
        engine_blocked.step(rain, dt, drainage_capacity_factor=0.1) # 90% blocked
        
    depth_clean = engine_clean.overland.h
    depth_blocked = engine_blocked.overland.h
    
    vol_clean = np.sum(depth_clean)
    vol_blocked = np.sum(depth_blocked)
    
    assert vol_blocked > vol_clean, f"Blocked drainage ({vol_blocked}) must result in higher surface water accumulation than clean ({vol_clean})."
