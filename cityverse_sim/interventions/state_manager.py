"""
cityverse_sim.interventions.state_manager
=========================================
Interactive city state manager supporting live mid-simulation user interventions
and dual execution engines (Numerical Physics Engine vs Learned World Model).
"""

from typing import Dict, Any, List, Tuple, Optional
import copy
import numpy as np
import torch

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.hydraulics.coupled_solver import CoupledHydraulicsEngine
from cityverse_sim.transportation.traffic_graph import DynamicRoadNetwork
from cityverse_sim.transportation.congestion import BPRCongestionModel
from cityverse_sim.transportation.routing import TrafficRouter
from cityverse_sim.infrastructure.electrical_grid import ElectricalGridManager
from cityverse_sim.emergency.dispatch import EmergencyDispatcher
from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel


class CityStateManager:
    """
    Manages the full state S(t) of MetroVerdia and allows dynamically injecting
    user actions A(t) at any point during a simulation run.
    """

    def __init__(
        self,
        city: Optional[MetroVerdiaCity] = None,
        world_model: Optional[UrbanWorldModel] = None,
        device: str = "cpu",
    ):
        self.city = city or MetroVerdiaCity("configs/city_spec.yaml")
        if not hasattr(self.city.grid, "dem") or self.city.grid.dem.max() == 0:
            self.city.build()
            
        self.device = device
        self.world_model = world_model
        if self.world_model is not None:
            self.world_model = self.world_model.to(self.device)
            self.world_model.eval()

        # Physical engine components
        self.hydro = CoupledHydraulicsEngine(self.city.grid, self.city.drainage, self.city.facilities)
        self.dyn_roads = DynamicRoadNetwork(self.city.roads, self.city.grid)
        self.congestion = BPRCongestionModel()
        self.router = TrafficRouter(self.dyn_roads, self.congestion)
        self.power = ElectricalGridManager(self.city.grid, self.city.facilities)
        self.dispatcher = EmergencyDispatcher(self.city.facilities, self.router)

        # Active state registers
        self.current_time_s = 0.0
        self.step_idx = 0
        
        # Current active controls/interventions A(t)
        self.active_rainfall_mm_hr = 0.0
        self.drainage_capacity_factor = 1.0
        self.pump_overrides: Dict[str, float] = {}
        self.manual_roadblocks: set = set() # Set of (u, v) blocked by emergency barriers
        
        # Static normalized DEM for neural model
        self.dem_normalized = torch.from_numpy(
            (self.city.grid.dem - self.city.grid.dem.min()) / max(1.0, (self.city.grid.dem.max() - self.city.grid.dem.min()))
        ).unsqueeze(0).unsqueeze(0).float().to(self.device) # (1, 1, H, W)
        
        # State history
        self.history_timestamps: List[float] = [0.0]
        self.history_scalars: List[Dict[str, Any]] = []

    def inject_intervention(self, intervention: Dict[str, Any]):
        """
        Applies a mid-simulation user intervention.
        Supported keys:
            - 'rainfall_mm_hr': float
            - 'drainage_capacity_factor': float in [0.0, 1.0]
            - 'pump_overrides': Dict[str, float] (e.g. {'PUMP-1': 1.0})
            - 'block_road': Tuple[str, str] (closes a specific road link)
            - 'unblock_road': Tuple[str, str] (reopens a road link)
            - 'flood_barrier': Dict with 'cells': List[(r, c)], 'height_m': float
        """
        if "rainfall_mm_hr" in intervention:
            self.active_rainfall_mm_hr = float(intervention["rainfall_mm_hr"])
        if "drainage_capacity_factor" in intervention:
            self.drainage_capacity_factor = float(np.clip(intervention["drainage_capacity_factor"], 0.0, 1.0))
        if "pump_overrides" in intervention:
            self.pump_overrides.update(intervention["pump_overrides"])
        if "block_road" in intervention:
            u, v = intervention["block_road"]
            self.manual_roadblocks.add((u, v))
            self.manual_roadblocks.add((v, u)) # Bidirectional
        if "unblock_road" in intervention:
            u, v = intervention["unblock_road"]
            self.manual_roadblocks.discard((u, v))
            self.manual_roadblocks.discard((v, u))
        if "flood_barrier" in intervention:
            # Raise local DEM to simulate physical flood wall / sandbag deployment
            barrier = intervention["flood_barrier"]
            h_add = float(barrier.get("height_m", 1.0))
            for r, c in barrier.get("cells", []):
                self.city.grid.dem[r, c] += h_add
                self.hydro.overland.dem[r, c] += h_add

    def step_physics(self, dt_s: float = 60.0) -> Dict[str, Any]:
        """Advances city state using high-fidelity numerical physics."""
        # 1. Advance hydraulics
        hydro_metrics = self.hydro.step(
            rainfall_rate_mm_hr=self.active_rainfall_mm_hr,
            dt_s=dt_s,
            drainage_capacity_factor=self.drainage_capacity_factor,
            pump_override=self.pump_overrides,
        )
        
        # 2. Update electrical grid status
        h_surface = self.hydro.overland.h
        power_status = self.power.update_grid_status(h_surface)
        
        # 3. Update road network and apply manual roadblocks
        self.dyn_roads.update_flood_conditions(h_surface)
        for u, v in self.manual_roadblocks:
            if self.dyn_roads.graph.has_edge(u, v):
                self.dyn_roads.graph[u][v]["is_passable_passenger"] = False
                self.dyn_roads.graph[u][v]["is_passable_emergency"] = False
                self.dyn_roads.graph[u][v]["flood_speed_mps"] = 0.0
                
        # 4. Update dynamic congestion travel times
        self.congestion.update_network_travel_times(self.dyn_roads.graph, {}, vehicle_type="emergency")
        
        # 5. Evaluate EMS Response Coverage
        sample_nodes = list(self.city.roads.graph.nodes())[::5]
        ems_cov = self.dispatcher.evaluate_emergency_coverage(sample_nodes)
        
        self.current_time_s += dt_s
        self.step_idx += 1
        
        snapshot = {
            "time_s": self.current_time_s,
            "engine": "physics",
            "rainfall_rate_mm_hr": self.active_rainfall_mm_hr,
            "flooded_area_km2_gt15cm": hydro_metrics["flooded_area_km2_gt15cm"],
            "flooded_area_km2_gt30cm": hydro_metrics["flooded_area_km2_gt30cm"],
            "max_water_depth_m": hydro_metrics["max_surface_depth_m"],
            "operational_substations": power_status["operational_substations"],
            "disabled_pumps": len(power_status["disabled_pumps"]),
            "closed_roads_passenger": self.dyn_roads.get_summary_metrics()["closed_edges_passenger"],
            "closed_roads_emergency": self.dyn_roads.get_summary_metrics()["closed_edges_emergency"],
            "ems_coverage_pct": ems_cov["coverage_pct"],
            "avg_ems_response_min": ems_cov["avg_response_time_min"],
        }
        self.history_timestamps.append(self.current_time_s)
        self.history_scalars.append(snapshot)
        return snapshot

    def step_world_model(self, dt_s: float = 300.0) -> Dict[str, Any]:
        """
        Advances city state in ~2 milliseconds using the trained neural World Model.
        """
        if self.world_model is None:
            raise RuntimeError("World Model not loaded in CityStateManager.")
            
        h_t = torch.from_numpy(self.hydro.overland.h).unsqueeze(0).unsqueeze(0).float().to(self.device)
        grid_in = torch.cat([h_t, self.dem_normalized], dim=1) # (1, 2, H, W)
        
        # Current scalar vector (11,)
        last_scalar = self.history_scalars[-1] if self.history_scalars else {}
        scalar_vec = torch.tensor([
            self.active_rainfall_mm_hr,
            self.drainage_capacity_factor,
            last_scalar.get("flooded_area_km2_gt15cm", 0.0),
            last_scalar.get("flooded_area_km2_gt30cm", 0.0),
            float(np.sum(self.hydro.overland.h) * self.city.grid.dx * self.city.grid.dy),
            float(last_scalar.get("closed_roads_passenger", 0)),
            float(last_scalar.get("closed_roads_emergency", 0)),
            float(last_scalar.get("operational_substations", 3)),
            float(last_scalar.get("disabled_pumps", 0)),
            float(last_scalar.get("ems_coverage_pct", 100.0)),
            float(last_scalar.get("avg_ems_response_min", 5.0)),
        ], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        p1 = self.pump_overrides.get("PUMP-1", 1.0)
        p2 = self.pump_overrides.get("PUMP-2", 1.0)
        action_vec = torch.tensor([
            self.active_rainfall_mm_hr,
            self.drainage_capacity_factor,
            p1,
            p2
        ], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            out = self.world_model(grid_in, scalar_vec, action_vec)
            
        pred_h = out["pred_grid_h"].squeeze().cpu().numpy()
        pred_scalars = out["pred_scalars"].squeeze().cpu().numpy()
        
        # Update surface water depth and dynamic road closures
        self.hydro.overland.h = pred_h.astype(np.float32)
        self.dyn_roads.update_flood_conditions(self.hydro.overland.h)
        road_summary = self.dyn_roads.get_summary_metrics()
        self.current_time_s += dt_s
        self.step_idx += 1
        
        snapshot = {
            "time_s": self.current_time_s,
            "engine": "world_model",
            "rainfall_rate_mm_hr": self.active_rainfall_mm_hr,
            "flooded_area_km2_gt15cm": float(np.sum(pred_h >= 0.15) * self.city.grid.dx * self.city.grid.dy / 1e6),
            "flooded_area_km2_gt30cm": float(np.sum(pred_h >= 0.30) * self.city.grid.dx * self.city.grid.dy / 1e6),
            "max_water_depth_m": float(np.max(pred_h)),
            "operational_substations": int(np.clip(np.round(pred_scalars[7]), 0, 3)),
            "disabled_pumps": int(np.clip(np.round(pred_scalars[8]), 0, 2)),
            "closed_roads_passenger": road_summary["closed_edges_passenger"],
            "closed_roads_emergency": road_summary["closed_edges_emergency"],
            "ems_coverage_pct": float(np.clip(pred_scalars[9], 0.0, 100.0)),
            "avg_ems_response_min": float(np.clip(pred_scalars[10], 1.0, 45.0)),
        }
        self.history_timestamps.append(self.current_time_s)
        self.history_scalars.append(snapshot)
        return snapshot

    def clone(self) -> 'CityStateManager':
        """Deep-copies city state for counterfactual branching."""
        cloned = CityStateManager(self.city, self.world_model, self.device)
        cloned.current_time_s = self.current_time_s
        cloned.step_idx = self.step_idx
        cloned.active_rainfall_mm_hr = self.active_rainfall_mm_hr
        cloned.drainage_capacity_factor = self.drainage_capacity_factor
        cloned.pump_overrides = copy.deepcopy(self.pump_overrides)
        cloned.manual_roadblocks = copy.deepcopy(self.manual_roadblocks)
        
        # Deep copy overland surface water
        cloned.hydro.overland.h = self.hydro.overland.h.copy()
        cloned.hydro.infiltration.cumulative_infiltrated_m = self.hydro.infiltration.cumulative_infiltrated_m.copy()
        
        # Copy history
        cloned.history_timestamps = list(self.history_timestamps)
        cloned.history_scalars = copy.deepcopy(self.history_scalars)
        return cloned

