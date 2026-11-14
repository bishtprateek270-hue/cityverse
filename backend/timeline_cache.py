"""
backend.timeline_cache
======================
High-performance timeline trajectory buffer and scenario execution engine.
Caches multi-step forward rollouts over time (e.g. 3-hour storm in 5-minute increments),
enabling smooth time-scrubbing, backward/forward stepping, and counterfactual comparisons.
"""

from typing import Dict, Any, List, Optional
import copy
import numpy as np

from cityverse_sim.interventions.state_manager import CityStateManager
from cityverse_sim.interventions.branch_engine import CounterfactualBranchEngine


class TimelineTrajectoryCache:
    """Stores full temporal simulation trajectories for timeline scrubbing."""

    def __init__(self, max_frames: int = 120):
        self.max_frames = max_frames
        self.branches: Dict[str, List[Dict[str, Any]]] = {
            "baseline": []
        }
        self.active_branch: str = "baseline"
        self.current_frame_idx: int = 0
        self.scenario_meta: Dict[str, Any] = {
            "rainfall_mm_hr": 120.0,
            "duration_hours": 3.0,
            "drainage_capacity_factor": 0.6,
            "engine_mode": "physics",
        }

    def clear(self, branch_id: Optional[str] = None):
        """Clears cached frames for a branch or all branches."""
        if branch_id:
            self.branches[branch_id] = []
        else:
            self.branches = {"baseline": []}
            self.current_frame_idx = 0

    def add_frame(self, branch_id: str, frame: Dict[str, Any]):
        """Appends a simulation frame to the branch trajectory."""
        if branch_id not in self.branches:
            self.branches[branch_id] = []
        self.branches[branch_id].append(frame)

    def get_frame(self, frame_idx: int, branch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a specific frame from the specified branch."""
        b = branch_id or self.active_branch
        frames = self.branches.get(b, [])
        if not frames:
            return None
        idx = int(np.clip(frame_idx, 0, len(frames) - 1))
        return frames[idx]

    def get_timeline_summary(self, branch_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns the sparkline profile and summary across the cached trajectory."""
        b = branch_id or self.active_branch
        frames = self.branches.get(b, [])
        if not frames:
            return {
                "branch_id": b,
                "total_frames": 0,
                "time_series": [],
                "flooded_area_km2_series": [],
                "ems_time_min_series": [],
                "closed_roads_series": []
            }

        return {
            "branch_id": b,
            "total_frames": len(frames),
            "scenario": self.scenario_meta,
            "time_series": [f["time_s"] for f in frames],
            "flooded_area_km2_series": [f["snapshot"].get("flooded_area_km2_gt15cm", 0.0) for f in frames],
            "ems_time_min_series": [f["snapshot"].get("avg_ems_response_min", 5.2) for f in frames],
            "closed_roads_series": [f["snapshot"].get("closed_roads_passenger", 0) for f in frames],
        }

    def run_full_scenario(
        self,
        state_mgr: CityStateManager,
        rainfall_mm_hr: float = 120.0,
        duration_hours: float = 3.0,
        drainage_capacity_factor: float = 0.6,
        pump_overrides: Optional[Dict[str, float]] = None,
        engine_mode: str = "world_model",
        branch_id: str = "baseline",
        step_dt_s: float = 300.0, # 5-minute increments
    ) -> Dict[str, Any]:
        """
        Executes a complete multi-step storm trajectory rollout and caches all frames.
        """
        self.scenario_meta = {
            "rainfall_mm_hr": rainfall_mm_hr,
            "duration_hours": duration_hours,
            "drainage_capacity_factor": drainage_capacity_factor,
            "pump_overrides": pump_overrides or {},
            "engine_mode": engine_mode,
        }
        
        # Reset state manager
        state_mgr.current_time_s = 0.0
        state_mgr.step_idx = 0
        state_mgr.hydro.reset()
        state_mgr.active_rainfall_mm_hr = rainfall_mm_hr
        state_mgr.drainage_capacity_factor = drainage_capacity_factor
        state_mgr.pump_overrides = pump_overrides or {}
        state_mgr.manual_roadblocks.clear()
        state_mgr.history_scalars.clear()

        total_sim_seconds = duration_hours * 3600.0
        n_steps = int(np.ceil(total_sim_seconds / step_dt_s))
        
        frames = []

        # Record Initial Frame T=0
        init_roads = state_mgr.dyn_roads.get_summary_metrics()
        init_snap = {
            "time_s": 0.0,
            "engine": engine_mode,
            "rainfall_rate_mm_hr": rainfall_mm_hr,
            "flooded_area_km2_gt15cm": 0.0,
            "flooded_area_km2_gt30cm": 0.0,
            "max_water_depth_m": 0.0,
            "operational_substations": 3,
            "disabled_pumps": 0,
            "closed_roads_passenger": init_roads["closed_edges_passenger"],
            "closed_roads_emergency": init_roads["closed_edges_emergency"],
            "ems_coverage_pct": 100.0,
            "avg_ems_response_min": 5.2,
        }
        
        frames.append({
            "step_idx": 0,
            "time_s": 0.0,
            "surface_depth_grid": np.zeros(state_mgr.city.grid.dem.shape, dtype=np.float32).tolist(),
            "snapshot": init_snap,
        })

        # Rollout loop
        for s in range(1, n_steps + 1):
            current_time = s * step_dt_s
            
            # Rainfall stops after duration_hours
            if current_time > total_sim_seconds:
                state_mgr.active_rainfall_mm_hr = 0.0

            h_prev = state_mgr.hydro.overland.h.copy()
            if engine_mode == "world_model" and state_mgr.world_model is not None:
                try:
                    snap = state_mgr.step_world_model(dt_s=step_dt_s)
                    # If neural checkpoint outputs degenerate zero depth during active rain, restore state and run exact physics
                    if snap.get("max_water_depth_m", 0.0) < 0.02 and rainfall_mm_hr > 10.0:
                        state_mgr.hydro.overland.h = h_prev
                        state_mgr.current_time_s -= step_dt_s
                        state_mgr.step_idx -= 1
                        if state_mgr.history_scalars:
                            state_mgr.history_scalars.pop()
                        snap = state_mgr.step_physics(dt_s=step_dt_s)
                except Exception:
                    state_mgr.hydro.overland.h = h_prev
                    snap = state_mgr.step_physics(dt_s=step_dt_s)
            else:
                snap = state_mgr.step_physics(dt_s=step_dt_s)

            h_surface = state_mgr.hydro.overland.h.copy()

            frame = {
                "step_idx": s,
                "time_s": current_time,
                "surface_depth_grid": h_surface.tolist(),
                "snapshot": snap,
            }
            frames.append(frame)

        self.branches[branch_id] = frames
        self.active_branch = branch_id
        self.current_frame_idx = 0

        return self.get_timeline_summary(branch_id)

