"""
cityverse_sim.real_world.calibration
====================================
Model calibration and data provenance validation protocol.
Ensures simulated predictions are never falsely claimed as real-world validated
without explicit historical gauge and satellite SAR validation.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np


class ModelProvenanceTier(str, Enum):
    """Data provenance classification tiers."""
    SYNTHETIC_SIMULATION = "SYNTHETIC_SIMULATION"
    REAL_WORLD_UNCALIBRATED = "REAL_WORLD_UNCALIBRATED"
    REAL_WORLD_CALIBRATED_BENCHMARK = "REAL_WORLD_CALIBRATED_BENCHMARK"


class ModelCalibrationProtocol:
    """
    Validates model predictions against historical real-world observations
    and assigns official provenance certificates.
    """

    def __init__(self, city_name: str = "MetroVerdia"):
        self.city_name = city_name
        self.tier = ModelProvenanceTier.SYNTHETIC_SIMULATION
        self.calibration_event: Optional[str] = None
        self.calibration_metrics: Dict[str, float] = {}

    def set_real_world_uncalibrated(self, real_region_name: str, dem_source: str, osm_source: str):
        """Marks dataset as real-world geography but uncalibrated."""
        self.city_name = real_region_name
        self.tier = ModelProvenanceTier.REAL_WORLD_UNCALIBRATED
        self.dem_source = dem_source
        self.osm_source = osm_source

    def calibrate_against_historical_event(
        self,
        event_name: str,
        observed_flood_extent_mask: np.ndarray,
        simulated_depth_grid: np.ndarray,
        depth_threshold_m: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Validates simulated inundation against real satellite SAR (e.g. Sentinel-1)
        or high-water survey observations.
        
        Metrics:
            - Critical Success Index (IoU)
            - False Alarm Rate (FAR)
            - Miss Rate
            - Inundation Area Bias (B = N_pred / N_obs)
        """
        obs_wet = observed_flood_extent_mask.astype(bool)
        pred_wet = simulated_depth_grid >= depth_threshold_m
        
        tp = np.logical_and(pred_wet, obs_wet).sum()
        fp = np.logical_and(pred_wet, ~obs_wet).sum()
        fn = np.logical_and(~pred_wet, obs_wet).sum()
        
        iou = float((tp + 1e-4) / (tp + fp + fn + 1e-4))
        far = float(fp / max(1, tp + fp))
        miss_rate = float(fn / max(1, tp + fn))
        bias = float((tp + fp) / max(1, tp + fn))
        
        metrics = {
            "critical_success_index_iou": iou,
            "false_alarm_rate": far,
            "miss_rate": miss_rate,
            "inundation_bias": bias,
        }
        
        # Calibration is certified only if IoU >= 0.70 and Bias in [0.8, 1.25]
        is_certified = (iou >= 0.70) and (0.80 <= bias <= 1.25)
        
        if is_certified:
            self.tier = ModelProvenanceTier.REAL_WORLD_CALIBRATED_BENCHMARK
            self.calibration_event = event_name
            self.calibration_metrics = metrics
            
        return {
            "event_name": event_name,
            "provenance_tier": self.tier.value,
            "is_certified": is_certified,
            "metrics": metrics,
        }

    def get_provenance_manifest(self) -> Dict[str, Any]:
        """Returns official provenance declaration."""
        return {
            "city_name": self.city_name,
            "provenance_tier": self.tier.value,
            "is_real_world_calibrated": self.tier == ModelProvenanceTier.REAL_WORLD_CALIBRATED_BENCHMARK,
            "calibration_event": self.calibration_event,
            "calibration_metrics": self.calibration_metrics,
            "disclaimer": (
                "Predictions marked as SYNTHETIC or UNCALIBRATED are for computational simulation "
                "and disaster response scenario planning. They must not be used as certified real-world "
                "safety-critical evacuation directives without independent local hydrologic validation."
            )
        }

