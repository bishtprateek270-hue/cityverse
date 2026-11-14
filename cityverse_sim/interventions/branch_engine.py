"""
cityverse_sim.interventions.branch_engine
=========================================
Counterfactual scenario branching engine for CityVerse.
Manages scenario forks, parallel branch rollouts, and multi-system diff metrics.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from cityverse_sim.interventions.state_manager import CityStateManager


class CounterfactualBranchEngine:
    """
    Orchestrates alternative simulation branches:
    Allows user to fork reality at time t_fork, apply alternative interventions,
    and compare the resulting trajectories.
    """

    def __init__(self, root_state: CityStateManager):
        self.branches: Dict[str, CityStateManager] = {
            "baseline": root_state
        }
        self.branch_lineage: Dict[str, Dict[str, Any]] = {
            "baseline": {"parent": None, "fork_time_s": 0.0, "interventions": {}}
        }

    def fork_branch(
        self,
        new_branch_id: str,
        source_branch_id: str = "baseline",
        intervention: Optional[Dict[str, Any]] = None
    ) -> CityStateManager:
        """
        Forks an existing branch at its current timestep, applies interventions,
        and registers the new branch.
        """
        if source_branch_id not in self.branches:
            raise KeyError(f"Source branch '{source_branch_id}' does not exist.")
            
        source_state = self.branches[source_branch_id]
        forked_state = source_state.clone()
        
        if intervention:
            forked_state.inject_intervention(intervention)
            
        self.branches[new_branch_id] = forked_state
        self.branch_lineage[new_branch_id] = {
            "parent": source_branch_id,
            "fork_time_s": source_state.current_time_s,
            "interventions": intervention or {},
        }
        return forked_state

    def rollout_branches(
        self,
        branch_ids: List[str],
        duration_s: float = 1800.0, # 30 minutes forward
        dt_s: float = 60.0,
        use_world_model: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Advances designated branches simultaneously by duration_s seconds.
        """
        results = {}
        n_steps = int(np.ceil(duration_s / dt_s))
        
        for bid in branch_ids:
            state = self.branches[bid]
            snapshots = []
            for _ in range(n_steps):
                if use_world_model:
                    snap = state.step_world_model(dt_s=dt_s)
                else:
                    snap = state.step_physics(dt_s=dt_s)
                snapshots.append(snap)
            results[bid] = snapshots
            
        return results

    def compare_branches(
        self,
        branch_a_id: str,
        branch_b_id: str
    ) -> Dict[str, Any]:
        """
        Computes detailed quantitative diff between Branch A and Branch B at current state.
        
        Returns:
            Dict containing:
                delta_flooded_area_km2: Area_B - Area_A (negative means B reduced flood)
                delta_max_depth_m: Max_B - Max_A
                delta_avg_ems_time_min: EMS_B - EMS_A
                delta_operational_substations: Subs_B - Subs_A
                delta_closed_roads_passenger: Roads_B - Roads_A
                spatial_depth_diff_grid: 2D array of (h_B - h_A)
                is_branch_b_better: bool (overall mitigation effectiveness)
        """
        if branch_a_id not in self.branches or branch_b_id not in self.branches:
            raise KeyError("Specified branches do not exist.")
            
        state_a = self.branches[branch_a_id]
        state_b = self.branches[branch_b_id]
        
        h_a = state_a.hydro.overland.h
        h_b = state_b.hydro.overland.h
        
        # Spatial difference matrix
        h_diff = (h_b - h_a).astype(np.float32)
        
        snap_a = state_a.history_scalars[-1] if state_a.history_scalars else {}
        snap_b = state_b.history_scalars[-1] if state_b.history_scalars else {}
        
        delta_flood_area = snap_b.get("flooded_area_km2_gt15cm", 0.0) - snap_a.get("flooded_area_km2_gt15cm", 0.0)
        delta_ems_time = snap_b.get("avg_ems_response_min", 0.0) - snap_a.get("avg_ems_response_min", 0.0)
        delta_substations = snap_b.get("operational_substations", 3) - snap_a.get("operational_substations", 3)
        delta_roads = snap_b.get("closed_roads_passenger", 0) - snap_a.get("closed_roads_passenger", 0)
        
        mean_reduction = float(-np.mean(h_diff))
        # Branch B is effective if it reduced surface water depth or area, or saved substations / reduced EMS delay
        is_effective = (mean_reduction > 1e-4) or (delta_flood_area < 0.0) or (delta_ems_time < -0.1) or (delta_substations > 0)
        
        return {
            "branch_a": branch_a_id,
            "branch_b": branch_b_id,
            "time_s": state_a.current_time_s,
            "delta_flooded_area_km2": float(delta_flood_area),
            "delta_avg_ems_time_min": float(delta_ems_time),
            "delta_operational_substations": int(delta_substations),
            "delta_closed_roads_passenger": int(delta_roads),
            "mean_depth_reduction_m": mean_reduction,
            "spatial_depth_diff_grid": h_diff,
            "is_branch_b_better": bool(is_effective),
        }
