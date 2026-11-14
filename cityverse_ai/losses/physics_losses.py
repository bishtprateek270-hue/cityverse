"""
cityverse_ai.losses.physics_losses
==================================
Physics-informed loss functions for urban flood World Model training.
Combines weighted depth MAE, soft Critical Success Index (IoU),
system scalar loss, and mass conservation regularization.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedFloodLoss(nn.Module):
    """
    Multi-objective loss function enforcing hydrological consistency:
        L_total = L_mae + lambda_iou * L_soft_iou + lambda_mass * L_mass_conservation + lambda_scalar * L_scalar
    """

    def __init__(
        self,
        weight_iou: float = 1.5,
        weight_mass: float = 2.0,
        weight_scalars: float = 0.5,
        flood_threshold_m: float = 0.15,
    ):
        super().__init__()
        self.w_iou = float(weight_iou)
        self.w_mass = float(weight_mass)
        self.w_scalars = float(weight_scalars)
        self.flood_thresh = float(flood_threshold_m)

    def forward(
        self,
        pred_h: torch.Tensor,        # (B, 1, H, W)
        target_h: torch.Tensor,      # (B, 1, H, W)
        curr_h: torch.Tensor,        # (B, 1, H, W)
        actions: torch.Tensor,       # (B, 4) [rain_rate, alpha_drain, p1, p2]
        pred_scalars: torch.Tensor,  # (B, S_dim)
        target_scalars: torch.Tensor,# (B, S_dim)
        pred_roads: Optional[torch.Tensor] = None,  # (B, E)
        target_roads: Optional[torch.Tensor] = None,# (B, E)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite loss and individual metric breakdown.
        """
        # 1. Weighted Depth MAE
        # Penalize errors in flooded cells (h > 0.05m) more heavily than bone-dry cells
        weight_mask = 1.0 + 4.0 * (target_h > 0.05).float()
        l_mae = torch.mean(weight_mask * torch.abs(pred_h - target_h))
        
        # 2. Soft-IoU / Critical Success Index Loss on inundated areas (h > 0.15m)
        tau = 0.05 # temperature for sigmoid approximation
        pred_wet = torch.sigmoid((pred_h - self.flood_thresh) / tau)
        target_wet = (target_h >= self.flood_thresh).float()
        
        intersection = torch.sum(pred_wet * target_wet, dim=(1, 2, 3))
        union = torch.sum(pred_wet + target_wet - (pred_wet * target_wet), dim=(1, 2, 3))
        soft_iou = (intersection + 1e-4) / (union + 1e-4)
        l_iou = torch.mean(1.0 - soft_iou)
        
        # 3. Physics Mass Conservation Penalty
        pred_delta_vol = torch.mean(pred_h - curr_h, dim=(1, 2, 3))
        target_delta_vol = torch.mean(target_h - curr_h, dim=(1, 2, 3))
        l_mass = torch.mean(torch.abs(pred_delta_vol - target_delta_vol))
        
        # 4. System Scalars Loss (Huber Loss with canonical scale normalization)
        scalar_scales = torch.tensor(
            [100.0, 1.0, 5.0, 5.0, 50000.0, 100.0, 50.0, 3.0, 2.0, 100.0, 10.0],
            device=pred_scalars.device,
            dtype=pred_scalars.dtype
        )
        l_scalars = F.smooth_l1_loss(pred_scalars / scalar_scales, target_scalars / scalar_scales)
        
        # 5. Road Depth Loss
        l_roads = torch.tensor(0.0, device=pred_h.device)
        if pred_roads is not None and target_roads is not None:
            l_roads = F.smooth_l1_loss(pred_roads, target_roads)
        
        # Total composite loss
        total_loss = l_mae + self.w_iou * l_iou + self.w_mass * l_mass + self.w_scalars * l_scalars + 0.5 * l_roads
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_mae": float(l_mae.item()),
            "loss_iou": float(l_iou.item()),
            "loss_mass": float(l_mass.item()),
            "loss_scalars": float(l_scalars.item()),
            "loss_roads": float(l_roads.item()),
            "mean_iou": float(torch.mean(soft_iou).item()),
        }
        
        return total_loss, metrics

