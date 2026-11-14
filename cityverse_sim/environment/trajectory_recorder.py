"""
cityverse_sim.environment.trajectory_recorder
=============================================
Orchestrates coupled simulation runs and records high-dimensional spatiotemporal
state-action trajectories (S_0, A_0, S_1, A_1, ..., S_T) for learned World Model training.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.hydraulics.coupled_solver import CoupledHydraulicsEngine
from cityverse_sim.transportation.traffic_graph import DynamicRoadNetwork
from cityverse_sim.transportation.congestion import BPRCongestionModel
from cityverse_sim.transportation.routing import TrafficRouter
from cityverse_sim.infrastructure.electrical_grid import ElectricalGridManager
from cityverse_sim.emergency.dispatch import EmergencyDispatcher
from cityverse_sim.environment.rainfall_generator import RainfallHyetograph


class TrajectoryRecorder:
    """
    Executes a complete city simulation trajectory and records state-action frames.
    """

    def __init__(self, city: MetroVerdiaCity):
        self.city = city
        self.grid = city.grid
        self.roads = city.roads
        self.drainage = city.drainage
        self.facilities = city.facilities
        
        # Instantiate subsystems
        self.hydro = CoupledHydraulicsEngine(self.grid, self.drainage, self.facilities)
        self.dyn_roads = DynamicRoadNetwork(self.roads, self.grid)
        self.congestion = BPRCongestionModel()
        self.router = TrafficRouter(self.dyn_roads, self.congestion)
        self.power = ElectricalGridManager(self.grid, self.facilities)
        self.dispatcher = EmergencyDispatcher(self.facilities, self.router)
        
        # Canonical ordered list of road edge IDs
        self.road_edges = list(self.roads.graph.edges())
        # Sample subset of nodes for fast EMS coverage tracking
        self.ems_sample_nodes = list(self.roads.graph.nodes())[::4] # Sample every 4th node for efficiency

    def run_trajectory(
        self,
        hyetograph: RainfallHyetograph,
        drainage_capacity_factor: float = 1.0,
        pump_override: Optional[Dict[str, float]] = None,
        dt_sim_s: float = 60.0,
        record_interval_s: float = 300.0, # Record every 5 minutes
    ) -> Dict[str, Any]:
        """
        Runs the simulation from t=0 to t=hyetograph.duration_s, recording frames.
        
        Returns:
            Dict of arrays:
                grid_depths: (T, H, W) float32
                road_depths: (T, E) float32
                road_speeds: (T, E) float32
                road_passable: (T, E) uint8
                system_scalars: (T, S_dim) float32
                actions: (T, A_dim) float32
                metadata: Dict
        """
        self.hydro.reset()
        
        total_duration_s = hyetograph.duration_s
        n_frames = int(np.floor(total_duration_s / record_interval_s)) + 1
        
        T_steps = []
        grid_frames = []
        road_depth_frames = []
        road_speed_frames = []
        road_passable_frames = []
        scalar_frames = []
        action_frames = []
        
        current_time_s = 0.0
        next_record_time_s = 0.0
        frame_idx = 0
        
        while current_time_s <= total_duration_s + 1e-3:
            # 1. External Action at current time
            rain_rate = hyetograph.get_intensity(current_time_s)
            
            # Check if this is a snapshot time
            if current_time_s >= next_record_time_s - 1e-3:
                # A. Sample Surface Water
                h_surface = self.hydro.overland.h.copy()
                
                # B. Update Electrical Grid Status
                power_status = self.power.update_grid_status(h_surface)
                
                # C. Update Road Network Status
                self.dyn_roads.update_flood_conditions(h_surface)
                self.congestion.update_network_travel_times(
                    self.dyn_roads.graph, {}, vehicle_type="emergency"
                )
                
                # Extract road vectors
                r_depths = np.zeros(len(self.road_edges), dtype=np.float32)
                r_speeds = np.zeros(len(self.road_edges), dtype=np.float32)
                r_passable = np.zeros(len(self.road_edges), dtype=np.uint8)
                
                for idx, (u, v) in enumerate(self.road_edges):
                    edata = self.dyn_roads.graph[u][v]
                    r_depths[idx] = edata.get("current_water_depth_m", 0.0)
                    r_speeds[idx] = edata.get("current_speed_mps", 0.0)
                    r_passable[idx] = 1 if edata.get("is_passable_passenger", True) else 0
                    
                # D. Evaluate EMS Response Coverage
                ems_cov = self.dispatcher.evaluate_emergency_coverage(self.ems_sample_nodes)
                
                # E. Assemble System Scalar State
                road_metrics = self.dyn_roads.get_summary_metrics()
                flooded_15cm = float(np.sum(h_surface >= 0.15) * self.grid.dx * self.grid.dy / 1e6)
                flooded_30cm = float(np.sum(h_surface >= 0.30) * self.grid.dx * self.grid.dy / 1e6)
                total_water_vol = float(np.sum(h_surface) * self.grid.dx * self.grid.dy)
                
                system_scalar = np.array([
                    rain_rate,
                    drainage_capacity_factor,
                    flooded_15cm,
                    flooded_30cm,
                    total_water_vol,
                    float(road_metrics["closed_edges_passenger"]),
                    float(road_metrics["closed_edges_emergency"]),
                    float(power_status["operational_substations"]),
                    float(len(power_status["disabled_pumps"])),
                    float(ems_cov["coverage_pct"]),
                    float(ems_cov["avg_response_time_min"]),
                ], dtype=np.float32)
                
                # F. Assemble Action Vector A(t)
                p1_throttle = 1.0 if "PUMP-1" not in power_status["disabled_pumps"] else 0.0
                p2_throttle = 1.0 if "PUMP-2" not in power_status["disabled_pumps"] else 0.0
                if pump_override:
                    p1_throttle = min(p1_throttle, pump_override.get("PUMP-1", 1.0))
                    p2_throttle = min(p2_throttle, pump_override.get("PUMP-2", 1.0))
                    
                action = np.array([
                    rain_rate,
                    drainage_capacity_factor,
                    p1_throttle,
                    p2_throttle,
                ], dtype=np.float32)
                
                # Store frame
                T_steps.append(current_time_s)
                grid_frames.append(h_surface)
                road_depth_frames.append(r_depths)
                road_speed_frames.append(r_speeds)
                road_passable_frames.append(r_passable)
                scalar_frames.append(system_scalar)
                action_frames.append(action)
                
                next_record_time_s += record_interval_s
                frame_idx += 1
                
            # Advance simulation step
            self.hydro.step(
                rainfall_rate_mm_hr=rain_rate,
                dt_s=dt_sim_s,
                drainage_capacity_factor=drainage_capacity_factor,
                pump_override=pump_override,
            )
            current_time_s += dt_sim_s

        return {
            "time_steps_s": np.array(T_steps, dtype=np.float32),
            "grid_depths": np.array(grid_frames, dtype=np.float32),        # (T, H, W)
            "road_depths": np.array(road_depth_frames, dtype=np.float32),  # (T, E)
            "road_speeds": np.array(road_speed_frames, dtype=np.float32),  # (T, E)
            "road_passable": np.array(road_passable_frames, dtype=np.uint8),# (T, E)
            "system_scalars": np.array(scalar_frames, dtype=np.float32),   # (T, 11)
            "actions": np.array(action_frames, dtype=np.float32),          # (T, 4)
            "metadata": {
                "peak_intensity_mm_hr": hyetograph.peak_intensity,
                "duration_hours": hyetograph.duration_hr,
                "profile_type": hyetograph.profile_type,
                "drainage_capacity_factor": drainage_capacity_factor,
                "total_frames": len(T_steps),
            }
        }

    @staticmethod
    def save_trajectory(trajectory_data: Dict[str, Any], filepath: str):
        """Saves trajectory to compressed .npz archive."""
        out_file = Path(filepath)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            out_file,
            time_steps_s=trajectory_data["time_steps_s"],
            grid_depths=trajectory_data["grid_depths"],
            road_depths=trajectory_data["road_depths"],
            road_speeds=trajectory_data["road_speeds"],
            road_passable=trajectory_data["road_passable"],
            system_scalars=trajectory_data["system_scalars"],
            actions=trajectory_data["actions"],
        )

