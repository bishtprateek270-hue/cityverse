"""
cityverse_sim.hydraulics.infiltration
====================================
Spatially distributed soil infiltration and effective rainfall models.
Implements the Horton Infiltration formulation with supply limitation
and land-cover imperviousness partitioning.
"""

from typing import Tuple
import numpy as np


class HortonInfiltration:
    """
    Spatially distributed Horton infiltration model.
    Governing equation for infiltration capacity:
        f_cap(t) = f_c + (f_0 - f_c) * exp(-k_d * t)
    Where:
        f_0 = initial infiltration capacity (mm/hr)
        f_c = ultimate saturated capacity (mm/hr)
        k_d = decay rate coefficient (1/hr)
    """

    def __init__(
        self,
        f0: np.ndarray,
        fc: np.ndarray,
        decay_k: np.ndarray,
        impervious_fraction: np.ndarray
    ):
        self.f0 = f0.astype(np.float32)
        self.fc = fc.astype(np.float32)
        self.decay_k = decay_k.astype(np.float32)
        self.impervious_fraction = np.clip(impervious_fraction, 0.0, 1.0).astype(np.float32)
        
        self.shape = f0.shape
        self.cumulative_infiltrated_m = np.zeros(self.shape, dtype=np.float32)
        self.elapsed_wetting_time_s = np.zeros(self.shape, dtype=np.float32)

    def reset(self):
        """Reset infiltration states to initial antecedent condition."""
        self.cumulative_infiltrated_m.fill(0.0)
        self.elapsed_wetting_time_s.fill(0.0)

    def step(
        self,
        rainfall_rate_mm_hr: float,
        surface_depth_m: np.ndarray,
        dt_s: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance infiltration by dt_s seconds.
        
        Args:
            rainfall_rate_mm_hr: Gross precipitation rate [mm/hr]
            surface_depth_m: (H, W) surface water depth on grid [m]
            dt_s: Time step in seconds
            
        Returns:
            infiltrated_depth_m: (H, W) depth infiltrated during dt [m]
            net_surface_rate_m_s: (H, W) net rate of water added to surface [m/s]
                                  (positive = net water accumulation, negative = water soaked up from puddle)
        """
        gross_rain_m = (rainfall_rate_mm_hr / 1000.0 / 3600.0) * dt_s
        gross_rain_grid = np.full(self.shape, gross_rain_m, dtype=np.float32)
        
        # Advance wetting time where water or rain is present
        has_water = (surface_depth_m > 1e-4) | (gross_rain_m > 1e-6)
        self.elapsed_wetting_time_s[has_water] += dt_s
        t_hr = self.elapsed_wetting_time_s / 3600.0
        
        # Potential infiltration capacity rate [m/s]
        f_cap_mm_hr = self.fc + (self.f0 - self.fc) * np.exp(-self.decay_k * t_hr)
        f_cap_m_s = (f_cap_mm_hr / 1000.0) / 3600.0
        
        # Pervious fraction
        pervious_frac = 1.0 - self.impervious_fraction
        
        # Maximum potential infiltration depth in this dt [m]
        potential_infil_depth_m = f_cap_m_s * dt_s * pervious_frac
        
        # Supply available: rain falling on pervious fraction + existing surface water on pervious fraction
        available_supply_m = (gross_rain_grid * pervious_frac) + surface_depth_m
        
        # Actual infiltration is bounded by supply and potential
        infiltrated_depth_m = np.minimum(potential_infil_depth_m, available_supply_m)
        infiltrated_depth_m = np.maximum(0.0, infiltrated_depth_m)
        
        self.cumulative_infiltrated_m += infiltrated_depth_m
        
        # Net depth added to surface: gross rainfall minus total infiltrated depth
        # Since rain on impervious fraction is not infiltrated, it automatically stays on the surface
        net_surface_depth_m = gross_rain_grid - infiltrated_depth_m
        net_surface_rate_m_s = net_surface_depth_m / max(dt_s, 1e-6)
        
        return infiltrated_depth_m, net_surface_rate_m_s
