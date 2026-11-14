"""
cityverse_sim.environment.rainfall_generator
============================================
Synthetic hyetograph generator for storm event simulation.
Implements the Chicago Design Storm method and Intensity-Duration-Frequency (IDF)
curves to produce realistic temporal precipitation profiles.
"""

from typing import Dict, Any, Tuple, Callable
import numpy as np


class RainfallHyetograph:
    """
    Generates temporal rainfall intensity profiles I(t) in mm/hr.
    """

    def __init__(
        self,
        peak_intensity_mm_hr: float = 80.0,
        duration_hours: float = 3.0,
        peak_ratio_r: float = 0.375, # Skew factor: 0.375 means peak at 37.5% of duration
        profile_type: str = "chicago", # "chicago", "triangular", "block"
    ):
        self.peak_intensity = float(peak_intensity_mm_hr)
        self.duration_hr = float(duration_hours)
        self.duration_s = self.duration_hr * 3600.0
        self.r = float(np.clip(peak_ratio_r, 0.1, 0.9))
        self.profile_type = profile_type
        
        # IDF empirical parameters: i(t) = a / (t_min + b)^c
        # Calibrated so peak instantaneous intensity equals peak_intensity
        self.t_peak_s = self.duration_s * self.r
        self.b_param = 15.0 # minutes
        self.c_param = 0.80
        # Compute scaling parameter 'a'
        self.a_param = self.peak_intensity * ((self.b_param)**self.c_param)

    def get_intensity(self, time_s: float) -> float:
        """
        Returns instantaneous rainfall intensity in mm/hr at time_s.
        """
        if time_s < 0.0 or time_s > self.duration_s:
            return 0.0
            
        if self.profile_type == "block":
            # Constant uniform rainfall
            return self.peak_intensity * 0.70
            
        elif self.profile_type == "triangular":
            # Linear ramp up to peak, then ramp down
            if time_s <= self.t_peak_s:
                ratio = time_s / max(1.0, self.t_peak_s)
            else:
                ratio = (self.duration_s - time_s) / max(1.0, self.duration_s - self.t_peak_s)
            return float(self.peak_intensity * max(0.05, ratio))
            
        else: # Chicago Design Storm
            # Separate equations before and after peak
            if time_s <= self.t_peak_s:
                dt_min = (self.t_peak_s - time_s) / 60.0
                denom = ((dt_min / self.r) + self.b_param)**self.c_param
                denom_0 = ((dt_min / self.r) + self.b_param)**(self.c_param + 1.0)
                term1 = self.a_param / denom
                term2 = (self.a_param * (1.0 - self.c_param) * (dt_min / self.r)) / denom_0
                intensity = term1 - term2
            else:
                dt_min = (time_s - self.t_peak_s) / 60.0
                denom = ((dt_min / (1.0 - self.r)) + self.b_param)**self.c_param
                denom_0 = ((dt_min / (1.0 - self.r)) + self.b_param)**(self.c_param + 1.0)
                term1 = self.a_param / denom
                term2 = (self.a_param * (1.0 - self.c_param) * (dt_min / (1.0 - self.r))) / denom_0
                intensity = term1 - term2
                
            return float(np.clip(intensity, 2.0, self.peak_intensity))

    def generate_series(self, dt_s: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates full time-series array of (times_s, intensities_mm_hr).
        """
        n_steps = int(np.ceil(self.duration_s / dt_s)) + 1
        times = np.linspace(0, self.duration_s, n_steps)
        intensities = np.array([self.get_intensity(t) for t in times], dtype=np.float32)
        return times, intensities

    def get_total_depth_mm(self, dt_s: float = 60.0) -> float:
        """Calculates total cumulative precipitation depth in mm."""
        _, intensities = self.generate_series(dt_s=dt_s)
        # Numerical integration: mm/hr * (dt_s / 3600)
        return float(np.sum(intensities) * (dt_s / 3600.0))

