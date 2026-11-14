"""
cityverse_ai.uncertainty.estimator
==================================
Uncertainty Quantification and Ensemble Rollout Engine for CityVerse.
Computes epistemic predictive variance via Monte Carlo Dropout & Deep Ensembles,
generating confidence intervals [h_lower, h_upper], probabilistic road closures,
and multi-trajectory trajectory fan-charts.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel


class UncertaintyEstimator:
    """
    Computes spatial predictive uncertainty and trajectory confidence bounds.
    """

    def __init__(
        self,
        model: UrbanWorldModel,
        device: str = "cpu",
        dropout_rate: float = 0.15,
    ):
        self.model = model.to(device)
        self.device = device
        self.dropout_rate = float(dropout_rate)
        
        # Add spatial dropout layer for MC dropout sampling
        self.mc_dropout = nn.Dropout2d(p=self.dropout_rate)

    def predict_with_uncertainty(
        self,
        grid_state_t: torch.Tensor, # (1, 2, H, W)
        scalars_t: torch.Tensor,    # (1, 11)
        actions: torch.Tensor,      # (1, 4)
        n_samples: int = 15,
        confidence_level: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Runs N stochastic forward passes to compute predictive mean and epistemic variance.
        
        Returns:
            Dict containing:
                mean_depth: (H, W) float32
                std_depth: (H, W) float32 (epistemic uncertainty map)
                lower_bound_depth: (H, W) (e.g. 5th percentile)
                upper_bound_depth: (H, W) (e.g. 95th percentile)
                road_closure_probabilities: (E,) float32 (probability of edge being closed)
                mean_scalars: (11,) float32
                scalar_uncertainties: (11,) float32
        """
        self.model.train() # Enable stochastic dropout
        
        samples_h = []
        samples_scalars = []
        samples_road_depths = []
        
        grid_in = grid_state_t.to(self.device)
        scalars_in = scalars_t.to(self.device)
        actions_in = actions.to(self.device)
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Forward pass with stochastic latent perturbation
                out = self.model(grid_in, scalars_in, actions_in)
                
                # Apply MC dropout to the predicted depth delta
                perturbed_h = F.relu(
                    grid_in[:, 0:1, :, :] + self.mc_dropout(out["pred_delta_h"])
                )
                
                samples_h.append(perturbed_h.squeeze().cpu().numpy())
                samples_scalars.append(out["pred_scalars"].squeeze().cpu().numpy())
                samples_road_depths.append(out["pred_road_depth"].squeeze().cpu().numpy())
                
        self.model.eval()
        
        # Stack sample tensors
        arr_h = np.stack(samples_h, axis=0)               # (N, H, W)
        arr_scalars = np.stack(samples_scalars, axis=0)   # (N, 11)
        arr_roads = np.stack(samples_road_depths, axis=0) # (N, E)
        
        # Mean and Standard Deviation
        mean_h = np.mean(arr_h, axis=0).astype(np.float32)
        std_h = np.std(arr_h, axis=0).astype(np.float32)
        
        # Percentile bounds
        alpha = (1.0 - confidence_level) / 2.0
        lower_pct = alpha * 100.0
        upper_pct = (1.0 - alpha) * 100.0
        
        lower_h = np.percentile(arr_h, lower_pct, axis=0).astype(np.float32)
        upper_h = np.percentile(arr_h, upper_pct, axis=0).astype(np.float32)
        lower_h = np.maximum(0.0, lower_h)
        
        # Road closure probability: fraction of passes where edge depth >= 0.15m
        prob_road_closure = np.mean(arr_roads >= 0.15, axis=0).astype(np.float32)
        
        # Scalar statistics
        mean_scalars = np.mean(arr_scalars, axis=0).astype(np.float32)
        std_scalars = np.std(arr_scalars, axis=0).astype(np.float32)
        
        return {
            "mean_depth": mean_h,
            "std_depth": std_h,
            "lower_bound_depth": lower_h,
            "upper_bound_depth": upper_h,
            "road_closure_probabilities": prob_road_closure,
            "mean_scalars": mean_scalars,
            "std_scalars": std_scalars,
            "sample_count": n_samples,
            "confidence_level": confidence_level,
        }

    def generate_multi_trajectory_ensemble(
        self,
        initial_grid_h: torch.Tensor,  # (1, 1, H, W)
        dem: torch.Tensor,             # (1, 1, H, W)
        initial_scalars: torch.Tensor, # (1, 11)
        action_sequence: torch.Tensor, # (K, 4)
        n_ensemble_members: int = 5,
    ) -> Dict[str, Any]:
        """
        Rolls out multiple stochastic trajectory futures (fan-chart)
        from a single starting point.
        """
        K = action_sequence.shape[0]
        ensemble_runs = []
        
        for member_idx in range(n_ensemble_members):
            curr_h = initial_grid_h.clone().to(self.device)
            curr_dem = dem.to(self.device)
            curr_scalars = initial_scalars.clone().to(self.device)
            
            member_depth_frames = []
            member_flooded_areas = []
            member_ems_times = []
            
            with torch.no_grad():
                for k in range(K):
                    grid_in = torch.cat([curr_h, curr_dem], dim=1)
                    act = action_sequence[k:k+1].to(self.device)
                    
                    self.model.train()
                    out = self.model(grid_in, curr_scalars, act)
                    pred_h = F.relu(curr_h + self.mc_dropout(out["pred_delta_h"]))
                    pred_scalars = out["pred_scalars"]
                    
                    flooded_area = float(torch.sum(pred_h >= 0.15).item() * (50.0 * 50.0) / 1e6)
                    ems_time = float(pred_scalars[0, 10].item())
                    
                    member_depth_frames.append(pred_h.squeeze().cpu().numpy())
                    member_flooded_areas.append(flooded_area)
                    member_ems_times.append(ems_time)
                    
                    curr_h = pred_h
                    curr_scalars = pred_scalars
                    
            ensemble_runs.append({
                "member_id": member_idx,
                "flooded_areas": member_flooded_areas,
                "ems_times": member_ems_times,
            })
            
        self.model.eval()
        
        # Aggregate fan-chart percentiles across members
        areas_matrix = np.array([m["flooded_areas"] for m in ensemble_runs]) # (N, K)
        ems_matrix = np.array([m["ems_times"] for m in ensemble_runs])       # (N, K)
        
        return {
            "n_members": n_ensemble_members,
            "rollout_steps": K,
            "flooded_area_p10": np.percentile(areas_matrix, 10, axis=0).tolist(),
            "flooded_area_median": np.percentile(areas_matrix, 50, axis=0).tolist(),
            "flooded_area_p90": np.percentile(areas_matrix, 90, axis=0).tolist(),
            "ems_time_p10": np.percentile(ems_matrix, 10, axis=0).tolist(),
            "ems_time_median": np.percentile(ems_matrix, 50, axis=0).tolist(),
            "ems_time_p90": np.percentile(ems_matrix, 90, axis=0).tolist(),
            "ensemble_members": ensemble_runs,
        }

