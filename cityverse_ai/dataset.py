"""
cityverse_ai.dataset
====================
PyTorch Dataset loader for spatiotemporal urban trajectory archives.
Provides single-step (S_t, A_t -> S_{t+1}) and multi-step autoregressive rollout samples.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class UrbanTrajectoryDataset(Dataset):
    """
    Loads saved .npz urban trajectories and produces state-action transition pairs
    for training the learned World Model f_theta(S_t, A_t) -> S_{t+1}.
    """

    def __init__(
        self,
        trajectory_dir: str,
        rollout_horizon: int = 1, # Number of future steps to predict
        dem_path: str = "data/raw_city_gis/dem_elevation.npy",
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.rollout_horizon = int(rollout_horizon)
        
        # Load static DEM terrain elevation (H, W)
        if Path(dem_path).exists():
            self.dem = np.load(dem_path).astype(np.float32)
            self.dem_normalized = (self.dem - self.dem.min()) / max(1.0, (self.dem.max() - self.dem.min()))
        else:
            self.dem = np.zeros((120, 120), dtype=np.float32)
            self.dem_normalized = self.dem
            
        # Discover all .npz files in directory
        self.files = sorted(list(self.trajectory_dir.glob("*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz trajectory files found in {trajectory_dir}")
            
        # Index all valid (file_idx, start_t) windows
        self.samples: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self):
        """Precomputes start frame indices across all trajectories."""
        self.trajectories_cache = []
        for f_idx, fpath in enumerate(self.files):
            data = np.load(fpath)
            n_frames = len(data["grid_depths"])
            # Ensure we have at least rollout_horizon frames into future
            valid_starts = n_frames - self.rollout_horizon
            for t in range(valid_starts):
                self.samples.append((f_idx, t))
            # Cache data in memory if small, or keep lazily loaded
            self.trajectories_cache.append(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_idx, t = self.samples[idx]
        data = self.trajectories_cache[f_idx]
        
        # Current State S(t)
        # Grid channels: [surface_water_depth, normalized_dem] -> shape (2, H, W)
        h_t = data["grid_depths"][t] # (H, W)
        grid_input = np.stack([h_t, self.dem_normalized], axis=0) # (2, H, W)
        
        road_depth_t = data["road_depths"][t]     # (E,)
        road_speed_t = data["road_speeds"][t]     # (E,)
        scalars_t = data["system_scalars"][t]     # (11,)
        
        # Target Next States over rollout_horizon
        h_targets = []
        road_depth_targets = []
        road_speed_targets = []
        scalar_targets = []
        action_seq = []
        
        for k in range(self.rollout_horizon):
            t_next = t + 1 + k
            h_targets.append(data["grid_depths"][t_next])
            road_depth_targets.append(data["road_depths"][t_next])
            road_speed_targets.append(data["road_speeds"][t_next])
            scalar_targets.append(data["system_scalars"][t_next])
            action_seq.append(data["actions"][t + k]) # Action applied from t+k to t+k+1
            
        return {
            # Current State
            "grid_state_t": torch.from_numpy(grid_input).float(), # (2, H, W)
            "road_depth_t": torch.from_numpy(road_depth_t).float(),
            "road_speed_t": torch.from_numpy(road_speed_t).float(),
            "scalars_t": torch.from_numpy(scalars_t).float(),
            
            # Action Sequence
            "actions": torch.from_numpy(np.array(action_seq)).float(), # (K, 4)
            
            # Target Future States
            "target_grid_h": torch.from_numpy(np.array(h_targets)).float(),       # (K, H, W)
            "target_road_depth": torch.from_numpy(np.array(road_depth_targets)).float(), # (K, E)
            "target_road_speed": torch.from_numpy(np.array(road_speed_targets)).float(), # (K, E)
            "target_scalars": torch.from_numpy(np.array(scalar_targets)).float(),       # (K, 11)
        }

