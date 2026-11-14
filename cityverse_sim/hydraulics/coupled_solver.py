"""
cityverse_sim.hydraulics.coupled_solver
======================================
Dual-drainage coupling engine:
Bridges 2D overland surface inundation with 1D subsurface pipe network.
Models street curb inlet intake, conduit conveyance, pump lifting,
and hydraulic surcharge (manhole bubbling).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from cityverse_sim.city.grid import CityGrid
from cityverse_sim.city.network import DrainageNetwork, FacilityRegistry
from cityverse_sim.hydraulics.infiltration import HortonInfiltration
from cityverse_sim.hydraulics.overland_2d import Overland2DSolver


class CoupledHydraulicsEngine:
    """
    Coupled 2D Overland Surface + 1D Subsurface Drainage Simulation System.
    """

    def __init__(
        self,
        grid: CityGrid,
        drainage: DrainageNetwork,
        facilities: FacilityRegistry,
    ):
        self.grid = grid
        self.drainage = drainage
        self.facilities = facilities
        
        # 1. Soil infiltration subsystem
        self.infiltration = HortonInfiltration(
            f0=grid.horton_f0,
            fc=grid.horton_fc,
            decay_k=grid.horton_decay,
            impervious_fraction=grid.impervious_fraction,
        )
        
        # 2. 2D Overland solver subsystem
        self.overland = Overland2DSolver(
            dem=grid.dem,
            manning_n=grid.manning_n,
            dx=grid.dx,
            dy=grid.dy,
            river_mask=grid.river_mask,
        )
        
        # Precompute manhole grid cell locations
        self.manhole_cells = {}
        for nid, data in self.drainage.graph.nodes(data=True):
            r, c = self.grid.coord_to_cell(data["x"], data["y"])
            self.manhole_cells[nid] = (r, c)
            
        # Cumulative statistics
        self.total_rain_volume_m3 = 0.0
        self.total_infiltrated_volume_m3 = 0.0
        self.total_pumped_volume_m3 = 0.0

    def reset(self):
        """Reset all hydraulic states."""
        self.infiltration.reset()
        self.overland.reset()
        self.total_rain_volume_m3 = 0.0
        self.total_infiltrated_volume_m3 = 0.0
        self.total_pumped_volume_m3 = 0.0
        
        for _, data in self.drainage.graph.nodes(data=True):
            data["head_m"] = 0.0
            data["surcharge_m"] = 0.0
            
        for _, _, data in self.drainage.graph.edges(data=True):
            data["current_flow_m3_s"] = 0.0
            data["capacity_utilization"] = 0.0

    def step(
        self,
        rainfall_rate_mm_hr: float,
        dt_s: float,
        drainage_capacity_factor: float = 1.0,
        pump_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Advance coupled hydraulics by dt_s seconds.
        """
        drainage_alpha = float(np.clip(drainage_capacity_factor, 0.0, 1.0))
        
        # 1. Soil Infiltration Step
        infil_depth_m, net_surface_rate_m_s = self.infiltration.step(
            rainfall_rate_mm_hr=rainfall_rate_mm_hr,
            surface_depth_m=self.overland.h,
            dt_s=dt_s
        )
        
        infil_vol = float(np.sum(infil_depth_m) * self.overland.cell_area)
        self.total_infiltrated_volume_m3 += infil_vol
        
        gross_rain_vol = float((rainfall_rate_mm_hr / 1000.0 / 3600.0 * dt_s) * (self.grid.rows * self.grid.cols * self.overland.cell_area))
        self.total_rain_volume_m3 += gross_rain_vol
        
        # 2. Dual Drainage Exchange: Compute Curb Inlet Intake Rates
        net_drainage_rate_m_s = np.zeros(self.grid.dem.shape, dtype=np.float32)
        
        # Evaluate each junction/manhole
        for nid, data in self.drainage.graph.nodes(data=True):
            if data["node_type"] == "outfall":
                continue
                
            r, c = self.manhole_cells[nid]
            h_surface = self.overland.h[r, c]
            
            # Sum outgoing pipe capacities from this manhole
            out_edges = self.drainage.graph.out_edges(nid, data=True)
            max_pipe_cap = sum(d["capacity_m3_s"] for _, _, d in out_edges) * drainage_alpha
            
            if h_surface > 0.001:
                curb_len = data.get("inlet_curb_length_m", 1.2)
                if h_surface < 0.12:
                    q_inlet = 1.6 * curb_len * (h_surface**1.5)
                else:
                    inlet_area = curb_len * 0.15
                    q_inlet = 0.6 * inlet_area * np.sqrt(2.0 * 9.81 * h_surface)
                    
                q_actual = min(q_inlet, max_pipe_cap)
                max_vol_available = (h_surface * self.overland.cell_area) / dt_s
                q_actual = min(q_actual, max_vol_available)
                
                net_drainage_rate_m_s[r, c] += q_actual / self.overland.cell_area
                
                # Update pipe edge flow rates
                if out_edges:
                    flow_per_edge = q_actual / len(out_edges)
                    for _, _, edata in out_edges:
                        edata["current_flow_m3_s"] = flow_per_edge
                        edata["capacity_utilization"] = flow_per_edge / max(edata["capacity_m3_s"], 1e-4)
                        
        # 3. Pumping Station Operation
        pumps = self.facilities.get_by_category("pumping_stations")
        step_pumped_m3 = 0.0
        
        for pump in pumps:
            pid = pump["id"]
            nominal_cap = float(pump.get("capacity_m3_s", 10.0))
            power_source_id = pump.get("powered_by")
            
            sub_active = True
            if power_source_id and power_source_id in self.facilities.facilities:
                sub_active = self.facilities.facilities[power_source_id]["is_operational"]
                
            throttle = 1.0 if sub_active else 0.0
            if pump_override and pid in pump_override:
                throttle = min(throttle, float(pump_override[pid]))
                
            active_pump_cap = nominal_cap * throttle
            r, c = self.grid.coord_to_cell(pump["x"], pump["y"])
            local_depth = self.overland.h[r, c]
            if local_depth > 0.01 and active_pump_cap > 0:
                pump_vol = min(active_pump_cap * dt_s, local_depth * self.overland.cell_area)
                net_drainage_rate_m_s[r, c] += (pump_vol / dt_s) / self.overland.cell_area
                step_pumped_m3 += pump_vol
                
        self.total_pumped_volume_m3 += step_pumped_m3
        
        # 4. Advance 2D Overland Surface Solver
        mass_residual = self.overland.step(
            dt_s=dt_s,
            effective_runoff_rate_m_s=net_surface_rate_m_s,
            net_drainage_exchange_rate_m_s=net_drainage_rate_m_s,
        )
        
        # 5. Compute System Metrics
        surface_water_m = self.overland.h
        flooded_cells = (surface_water_m >= 0.15)
        critically_flooded_cells = (surface_water_m >= 0.30)
        
        total_surface_vol_m3 = float(np.sum(surface_water_m) * self.overland.cell_area)
        flooded_area_km2 = float(np.sum(flooded_cells) * self.overland.cell_area / 1e6)
        crit_flooded_area_km2 = float(np.sum(critically_flooded_cells) * self.overland.cell_area / 1e6)
        
        cumulative_drained = self.overland.cumulative_drainage_exchange_vol_m3
        cumulative_outflow = self.overland.cumulative_boundary_outflow_vol_m3
        
        return {
            "rainfall_rate_mm_hr": rainfall_rate_mm_hr,
            "dt_s": dt_s,
            "total_surface_water_vol_m3": total_surface_vol_m3,
            "max_surface_depth_m": float(np.max(surface_water_m)),
            "mean_surface_depth_m": float(np.mean(surface_water_m)),
            "flooded_area_km2_gt15cm": flooded_area_km2,
            "flooded_area_km2_gt30cm": crit_flooded_area_km2,
            "cumulative_rain_vol_m3": self.total_rain_volume_m3,
            "cumulative_infiltrated_vol_m3": self.total_infiltrated_volume_m3,
            "cumulative_drained_vol_m3": cumulative_drained,
            "cumulative_pumped_vol_m3": self.total_pumped_volume_m3,
            "boundary_discharge_vol_m3": cumulative_outflow,
            "mass_balance_residual_m3": mass_residual,
        }
