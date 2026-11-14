"""
cityverse_ai.models.spatio_temporal_world_model
===============================================
Learned Spatiotemporal World Model f_theta(S_t, A_t) -> S_{t+1}.
Combines a multi-scale residual U-Net neural operator for 2D surface inundation
with global latent MLP heads for road and cascading infrastructure predictions.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => LeakyReLU) * 2 with residual connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.res(x)


class UrbanWorldModel(nn.Module):
    """
    Learned dynamics model predicting next city state from current state and interventions:
        S_{t+1} = f_theta(S_t, A_t)
    """

    def __init__(
        self,
        in_grid_channels: int = 6, # h_t (1) + dem (1) + actions (4)
        base_channels: int = 32,
        scalar_dim: int = 11,
        action_dim: int = 4,
        num_road_edges: int = 464,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.action_dim = action_dim
        self.num_road_edges = num_road_edges
        
        # --- 2D Surface Inundation U-Net Encoder-Decoder ---
        c = base_channels
        self.inc = DoubleConv(in_grid_channels, c)         # (B, c, 120, 120)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c, c * 2)                          # (B, 2c, 60, 60)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c * 2, c * 4)                      # (B, 4c, 30, 30)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c * 4, c * 8)                      # (B, 8c, 15, 15)
        )
        
        # Bottleneck with dilated convolutions to expand receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, c * 8, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2) # 15 -> 30
        self.conv_up1 = DoubleConv(c * 8, c * 4)
        
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2) # 30 -> 60
        self.conv_up2 = DoubleConv(c * 4, c * 2)
        
        self.up3 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)     # 60 -> 120
        self.conv_up3 = DoubleConv(c * 2, c)
        
        # Surface delta head: predicts residual depth update Δh(t)
        self.out_head = nn.Conv2d(c, 1, kernel_size=1)
        
        # --- Scalar & Road Latent Head ---
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        latent_dim = (c * 8) + scalar_dim + action_dim
        
        self.scalar_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, scalar_dim)
        )
        
        self.road_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_road_edges)
        )

    def forward(
        self,
        grid_state_t: torch.Tensor, # (B, 2, H, W) [h_t, dem]
        scalars_t: torch.Tensor,    # (B, 11)
        actions: torch.Tensor,      # (B, 4)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass predicting S_{t+1}.
        
        Returns:
            Dict containing:
                pred_grid_h: (B, 1, H, W) next surface water depth
                pred_delta_h: (B, 1, H, W) predicted depth delta
                pred_scalars: (B, 11) next scalar indicators
                pred_road_depth: (B, E) predicted road depths
        """
        B, _, H, W = grid_state_t.shape
        h_t = grid_state_t[:, 0:1, :, :] # (B, 1, H, W)
        
        # Broadcast actions across spatial grid: (B, 4) -> (B, 4, H, W)
        action_spatial = actions.unsqueeze(-1).unsqueeze(-1).expand(B, self.action_dim, H, W)
        
        # Concatenate grid input channels: [h_t, dem, action_spatial] -> (B, 6, H, W)
        x_in = torch.cat([grid_state_t, action_spatial], dim=1)
        
        # U-Net Downward path
        x1 = self.inc(x_in)       # (B, c, 120, 120)
        x2 = self.down1(x1)       # (B, 2c, 60, 60)
        x3 = self.down2(x2)       # (B, 4c, 30, 30)
        x4 = self.down3(x3)       # (B, 8c, 15, 15)
        
        # Bottleneck
        b = self.bottleneck(x4)   # (B, 8c, 15, 15)
        
        # U-Net Upward path with skip connections
        u1 = self.up1(b)          # (B, 4c, 30, 30)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)         # (B, 2c, 60, 60)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)         # (B, c, 120, 120)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)
        
        # Predict depth delta Δh
        delta_h = self.out_head(u3)
        
        # Physical non-negative depth guarantee: h_{t+1} = max(0, h_t + Δh)
        pred_h = F.relu(h_t + delta_h)
        
        # Global latent vectors for scalar and road heads
        gap_feat = self.gap(b).view(B, -1) # (B, 8c)
        fused_latent = torch.cat([gap_feat, scalars_t, actions], dim=1)
        
        pred_scalars = self.scalar_head(fused_latent)
        pred_road_depth = F.relu(self.road_head(fused_latent))
        
        return {
            "pred_grid_h": pred_h,
            "pred_delta_h": delta_h,
            "pred_scalars": pred_scalars,
            "pred_road_depth": pred_road_depth,
        }

