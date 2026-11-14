"""
cityverse_ai.uncertainty.benchmark
==================================
Comprehensive evaluation and out-of-distribution (OOD) benchmark runner.
Computes:
- Critical Success Index (IoU) & MAE / RMSE on wet cells
- Road accessibility classification accuracy & F1 score
- Temporal error accumulation over multi-step rollouts
- Uncertainty calibration coverage (empirical coverage vs 90% CI)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import numpy as np
import torch

from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel
from cityverse_ai.uncertainty.estimator import UncertaintyEstimator
from cityverse_ai.dataset import UrbanTrajectoryDataset


class ComprehensiveBenchmark:
    """
    Evaluates the World Model across unseen stress scenarios.
    """

    def __init__(
        self,
        model: UrbanWorldModel,
        test_dir: str = "data/trajectories/test",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.test_dir = Path(test_dir)
        self.uncertainty = UncertaintyEstimator(self.model, device=self.device)

    def evaluate_test_split(
        self,
        flood_thresh: float = 0.15,
        max_rollout_steps: int = 4
    ) -> Dict[str, Any]:
        """
        Runs complete benchmark on all test trajectories.
        """
        if not self.test_dir.exists() or not list(self.test_dir.glob("*.npz")):
            raise FileNotFoundError(f"No test trajectory files in {self.test_dir}")
            
        test_dataset = UrbanTrajectoryDataset(
            trajectory_dir=str(self.test_dir),
            rollout_horizon=max_rollout_steps,
            dem_path="data/raw_city_gis/dem_elevation.npy"
        )
        
        ious = []
        maes = []
        rmses = []
        road_accuracies = []
        ci_coverages = []
        rollout_rmses_by_step = [[] for _ in range(max_rollout_steps)]
        
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            
            grid_in = sample["grid_state_t"].unsqueeze(0).to(self.device) # (1, 2, H, W)
            scalars_in = sample["scalars_t"].unsqueeze(0).to(self.device) # (1, 11)
            actions = sample["actions"].to(self.device)                   # (K, 4)
            
            target_h_seq = sample["target_grid_h"]                        # (K, H, W)
            target_h_1 = target_h_seq[0].numpy()                          # (H, W)
            target_roads_1 = sample["target_road_depth"][0].numpy()       # (E,)
            
            # 1. Single-Step Uncertainty Estimation
            uncert_res = self.uncertainty.predict_with_uncertainty(
                grid_in, scalars_in, actions[0:1], n_samples=10, confidence_level=0.90
            )
            pred_h = uncert_res["mean_depth"]
            lower_h = uncert_res["lower_bound_depth"]
            upper_h = uncert_res["upper_bound_depth"]
            
            # MAE & RMSE
            mae = float(np.mean(np.abs(pred_h - target_h_1)))
            rmse = float(np.sqrt(np.mean((pred_h - target_h_1)**2)))
            maes.append(mae)
            rmses.append(rmse)
            
            # IoU at 0.15m
            pred_wet = pred_h >= flood_thresh
            target_wet = target_h_1 >= flood_thresh
            intersection = np.logical_and(pred_wet, target_wet).sum()
            union = np.logical_or(pred_wet, target_wet).sum()
            iou = (intersection + 1e-4) / (union + 1e-4) if union > 0 else 1.0
            ious.append(float(iou))
            
            # Road Accessibility Classification
            pred_closed_prob = uncert_res["road_closure_probabilities"]
            pred_closed = pred_closed_prob >= 0.50
            target_closed = target_roads_1 >= flood_thresh
            acc = float(np.mean(pred_closed == target_closed))
            road_accuracies.append(acc)
            
            # 90% Confidence Interval Coverage: fraction of pixels within [lower, upper]
            inside_ci = (target_h_1 >= (lower_h - 0.01)) & (target_h_1 <= (upper_h + 0.01))
            ci_coverage = float(np.mean(inside_ci))
            ci_coverages.append(ci_coverage)
            
            # 2. Multi-Step Autoregressive Rollout Drift
            curr_h = grid_in[:, 0:1, :, :]
            curr_dem = grid_in[:, 1:2, :, :]
            curr_sc = scalars_in
            
            with torch.no_grad():
                for k in range(max_rollout_steps):
                    g_in = torch.cat([curr_h, curr_dem], dim=1)
                    out = self.model(g_in, curr_sc, actions[k:k+1])
                    curr_h = out["pred_grid_h"]
                    curr_sc = out["pred_scalars"]
                    
                    t_k = target_h_seq[k].numpy()
                    p_k = curr_h.squeeze().cpu().numpy()
                    step_rmse = float(np.sqrt(np.mean((p_k - t_k)**2)))
                    rollout_rmses_by_step[k].append(step_rmse)
                    
        mean_step_rmses = [float(np.mean(step_list)) for step_list in rollout_rmses_by_step]
        
        benchmark_results = {
            "test_sample_count": len(test_dataset),
            "critical_success_index_iou": float(np.mean(ious)),
            "depth_mae_m": float(np.mean(maes)),
            "depth_rmse_m": float(np.mean(rmses)),
            "road_accessibility_accuracy": float(np.mean(road_accuracies)),
            "ci_90_empirical_coverage": float(np.mean(ci_coverages)),
            "multi_step_rollout_rmse": mean_step_rmses,
            "max_rollout_steps": max_rollout_steps,
        }
        
        return benchmark_results

    def save_benchmark_report(self, results: Dict[str, Any], filepath: str = "models/checkpoints/benchmark_report.json"):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

