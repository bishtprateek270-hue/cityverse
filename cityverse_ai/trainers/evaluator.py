"""
cityverse_ai.trainers.evaluator
===============================
World Model benchmarking engine:
- Real-time inference latency benchmark (target < 30ms)
- Multi-step autoregressive rollout evaluation (K = 1 to 12 steps)
- Critical Success Index (IoU), RMSE, and mass drift tracking
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F

from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel


class WorldModelEvaluator:
    """
    Evaluates inference speed and multi-step autoregressive rollout accuracy.
    """

    def __init__(
        self,
        model: UrbanWorldModel,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def benchmark_inference_latency(
        self,
        n_warmup: int = 5,
        n_trials: int = 50,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Benchmarks inference latency per step in milliseconds.
        """
        dummy_grid = torch.zeros(batch_size, 2, 120, 120, device=self.device)
        dummy_scalars = torch.zeros(batch_size, 11, device=self.device)
        dummy_actions = torch.zeros(batch_size, 4, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(dummy_grid, dummy_scalars, dummy_actions)
                
        # Benchmark loop
        latencies_ms = []
        with torch.no_grad():
            for _ in range(n_trials):
                t0 = time.perf_counter()
                _ = self.model(dummy_grid, dummy_scalars, dummy_actions)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                
        mean_ms = float(np.mean(latencies_ms))
        p95_ms = float(np.percentile(latencies_ms, 95))
        min_ms = float(np.min(latencies_ms))
        
        return {
            "mean_latency_ms": mean_ms,
            "p95_latency_ms": p95_ms,
            "min_latency_ms": min_ms,
            "meets_target_30ms": bool(mean_ms < 30.0),
        }

    def evaluate_autoregressive_rollout(
        self,
        initial_grid_h: torch.Tensor,       # (1, 1, H, W)
        dem: torch.Tensor,                  # (1, 1, H, W)
        initial_scalars: torch.Tensor,      # (1, 11)
        action_sequence: torch.Tensor,      # (K, 4)
        target_grid_sequence: torch.Tensor, # (K, 1, H, W)
        flood_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Executes a recursive multi-step rollout:
            S_{t+1} = f_theta(S_t, A_t)
        Feeding predictions back as inputs.
        """
        K = action_sequence.shape[0]
        curr_h = initial_grid_h.to(self.device)
        curr_dem = dem.to(self.device)
        curr_scalars = initial_scalars.to(self.device)
        
        step_ious = []
        step_rmses = []
        predicted_h_frames = []
        
        with torch.no_grad():
            for k in range(K):
                # Form input grid: [h_t, dem] -> (1, 2, H, W)
                grid_in = torch.cat([curr_h, curr_dem], dim=1)
                act = action_sequence[k:k+1].to(self.device) # (1, 4)
                
                # Single-step prediction
                out = self.model(grid_in, curr_scalars, act)
                pred_h = out["pred_grid_h"]
                pred_scalars = out["pred_scalars"]
                
                # Compare against target ground-truth at step k
                target_h = target_grid_sequence[k:k+1].to(self.device)
                
                # RMSE
                rmse = float(torch.sqrt(torch.mean((pred_h - target_h)**2)).item())
                step_rmses.append(rmse)
                
                # Inundation IoU at threshold (0.15m)
                pred_wet = (pred_h >= flood_threshold).bool()
                target_wet = (target_h >= flood_threshold).bool()
                intersection = (pred_wet & target_wet).sum().item()
                union = (pred_wet | target_wet).sum().item()
                iou = (intersection + 1e-4) / (union + 1e-4) if union > 0 else 1.0
                step_ious.append(float(iou))
                
                predicted_h_frames.append(pred_h.cpu().numpy())
                
                # Recursive update for next step
                curr_h = pred_h
                curr_scalars = pred_scalars
                
        return {
            "rollout_steps": K,
            "mean_rmse": float(np.mean(step_rmses)),
            "mean_iou": float(np.mean(step_ious)),
            "step_rmses": step_rmses,
            "step_ious": step_ious,
            "predicted_h_frames": predicted_h_frames,
        }

